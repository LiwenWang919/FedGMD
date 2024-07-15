import collections
import os.path
import pickle
import queue
import warnings

import torch.cuda
import torch.multiprocessing as mlp
import flgo
import flgo.algorithm.fedavg as fedavg
import zmq
import time
import flgo.utils.fmodule
import threading
import numpy as np

def default_start_condition(server):
    return server.num_clients>=6

class Server(fedavg.Server):
    def __init__(self, option={}):
        self.test_data = None
        self.val_data = None
        self.train_data = None
        self.model = None
        # basic configuration
        self.task = option['task']
        self.eval_interval = option['eval_interval']
        self.num_parallels = option['num_parallels']
        # server calculator
        self.device = self.gv.apply_for_device() if not option['server_with_cpu'] else torch.device('cpu')
        self.calculator = self.TaskCalculator(self.device, optimizer_name=option['optimizer'])
        # hyper-parameters during training process
        self.num_rounds = option['num_rounds']
        self.num_steps = option['num_steps']
        self.num_epochs = option['num_epochs']
        self.proportion = option['proportion']
        self.decay_rate = option['learning_rate_decay']
        self.lr_scheduler_type = option['lr_scheduler']
        self.lr = option['learning_rate']
        self.learning_rate = option['learning_rate']
        self.sample_option = option['sample']
        self.aggregation_option = option['aggregate']
        # systemic option
        self.tolerance_for_latency = 999999
        # algorithm-dependent parameters
        self.algo_para = {}
        self.current_round = 1
        # all options
        self.option = option
        self.id = -1
        self._data_names = []
        self._exit = False
        self._avalability_timeout = 30
        self._communication_timeout = 1e10
        self.start_condition = default_start_condition

    def register_start_condition(self, f):
        assert callable(f)
        self.start_condition = f

    def set_availability_timeout(self, t:float):
        r"""
        Set the timeout of being unavailable for each client.

        Args:
            t (float): the time (secs)
        """
        self._avalability_timeout = t

    def set_communication_timeout(self, t:float):
        r"""
        Set the timeout of waiting for clients' responses

        Args:
            t (float): the time (secs)
        """
        self._communication_timeout = t

    def is_exit(self):
        r"""
        Return True when the server is going to close
        """
        self._lock_exit.acquire()
        res = self._exit
        self._lock_exit.release()
        return res

    def do_exit(self):
        r"""
        Set the flag of exit to be True
        """
        self._lock_exit.acquire()
        self._exit = True
        self._lock_exit.release()
        self.logger.info("Server was closed")
        return

    def add_buffer(self, x:dict):
        r"""
        Push x into network buffer (i.e. a queue)

        Args:
            x (dict): received packages from clients
        """
        self._lock_buffer.acquire()
        self._buffer.put(x)
        self._lock_buffer.release()
        return

    def clear_buffer(self):
        r"""
        Pop all elements in the buffer

        Return:
            res (dict): a dict contains pairs of (client_name, package) in buffer
        """
        self._lock_buffer.acquire()
        res = {}
        while not self._buffer.empty():
            d = self._buffer.get_nowait()
            res[d['name']] = d['package']
        self._lock_buffer.release()
        return res

    def size_buffer(self):
        r"""
        Return the number of current elements in buffer

        Return:
            buffer_size (int): the number of elements
        """
        self._lock_buffer.acquire()
        buffer_size = len(self._buffer)
        self._lock_buffer.release()
        return buffer_size

    def register(self):
        self.logger.info("Waiting for registrations...")
        while True:
            time.sleep(1)
            if self.if_start():
                self.logger.info("Start training...")
                break
        return

    def if_start(self):
        return self.start_condition(self)

    def register_handler(self, worker_id, client_id, received_pkg):
        valid_keys = ['num_steps', 'learning_rate', 'batch_size', 'momentum', 'weight_decay', 'num_epochs', 'optimizer']
        self._set_alive(received_pkg["name"], time.time())
        if received_pkg["name"] not in self.clients.keys():
            self.add_client(received_pkg["name"])
            l = len(self.clients)
            self.logger.info("%s joined in the federation. The number of clients is %i" % (received_pkg['name'], l))

            d = {"client_idx": l, 'port_send': self.port_send, 'port_recv': self.port_recv, 'port_alive':self.port_alive,
                 '__option__': {k: self.option[k] for k in valid_keys},'algo_para':self.algo_para}
            self.registrar.send_multipart([worker_id, client_id, pickle.dumps(d, pickle.DEFAULT_PROTOCOL)])
        else:
            self.logger.info("%s rebuilt the connection." % received_pkg['name'])
            self.registrar.send_pyobj({"client_idx": len(self.clients), 'port_send': self.port_send, 'port_recv': self.port_recv, 'port_alive':self.port_alive, '__option__': {k: self.option[k] for k in valid_keys}, 'algo_para':self.algo_para})

    def task_pusher_handler(self, worker_id, client_id):
        zipped_task = self._get_zipped_task()
        if zipped_task is None:
            self._read_zipped_task()
            zipped_task = self._get_zipped_task()
        self.task_pusher.send_multipart([worker_id, client_id, zipped_task])
        return

    def _listen(self):
        while not self.is_exit():
            events = dict(self._poller.poll(10000))
            if self.task_pusher in events and events[self.task_pusher]==zmq.POLLIN:
                worker_id, client_id, name, request = self.task_pusher.recv_multipart()
                if request==b'pull task':
                    try:
                        self.logger.info("Receive task pull request from {}".format(name))
                        t = threading.Thread(target=self.task_pusher_handler, args=(worker_id, client_id))
                        t.start()
                    except:
                        self.logger.info("Failed to handle task for %s" % client_id)
            if self.registrar in events and events[self.registrar]==zmq.POLLIN:
                worker_id = self.registrar.recv()
                client_id = self.registrar.recv()
                received_pkg = self.registrar.recv_pyobj()
                t = threading.Thread(target=self.register_handler, args=(worker_id, client_id, received_pkg))
                t.start()
            if self.receiver in events and events[self.receiver]==zmq.POLLIN:
                name = self.receiver.recv_string()
                package_msg = self.receiver.recv()
                package_size = len(package_msg) / 1024.0 / 1024.0
                d = self.receiver._deserialize(package_msg, pickle.loads)
                d['__size__'] = package_size
                if '__mtype__' in d and d['__mtype__'] == "close":
                    self.logger.info("{} was successfully closed.".format(name))
                else:
                    self.add_buffer({'name': name, 'package': d})
                    self.logger.info("Server Received package of size {}MB from {} at round {}".format(package_size, name, self.current_round))
            if self.alive_detector in events and events[self.alive_detector]==zmq.POLLIN:
                name, _ = self.alive_detector.recv_multipart()
                t = time.time()
                self.logger.debug(f"Client {name.decode('utf-8')} is alive at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                self._set_alive(name.decode('utf-8'), t)
                self.alive_detector.send(b"")
    @property
    def clients(self):
        self._lock_registration.acquire()
        res = self._clients
        self._lock_registration.release()
        return res

    @property
    def num_clients(self):
        return len(self.clients)

    def add_client(self, name):
        self._lock_registration.acquire()
        i = len(self._clients)
        self._clients[i] = name
        self._lock_registration.release()
        return

    def _set_alive(self, client_name:str, timestamp):
        self._lock_alive.acquire()
        self._alive_state[client_name] = timestamp
        self._lock_alive.release()

    @property
    def available_clients(self):
        crt_timestamp = time.time()
        self._lock_alive.acquire()
        avl_clients = [name for name in self._alive_state if crt_timestamp-self._alive_state[name]<=self._avalability_timeout]
        res = {k:v for k,v in self.clients.items() if v in avl_clients}
        self._lock_alive.release()
        return res

    def pack(self, client_id, mtype=0, *args, **kwargs):
        if mtype=='close':
            return {}
        else:
            return {'model': self.model}

    def _read_zipped_task(self, with_bmk=True, ignore_names=[]):
        task_path = self.option['task']
        task_name = os.path.basename(task_path)
        task_dir = os.path.dirname(os.path.abspath(task_path))
        task_zip = task_name + '.zip'
        if not os.path.exists(task_zip):
            flgo.zip_task(task_path, target_path=task_dir, with_bmk=with_bmk, ignore_names=ignore_names)
        if not hasattr(self, '_zipped_task'): self._zipped_task = []
        CHUNK_SIZE = 1024
        with open(os.path.join(task_dir, task_zip), 'rb') as inf:
            while True:
                chunk = inf.read(CHUNK_SIZE)
                self._zipped_task.append(chunk)
                if not chunk:
                    break

    def _get_zipped_task(self):
        if not hasattr(self, '_zipped_task'): return None
        return b"".join(self._zipped_task)

    def run(self, ip:str='*', port:str='5555', port_task:str='', protocol:str='tcp', no_zip_flags:list=['data.json', '__pycache__', 'checkpoint', 'record', 'log']):
        """
        Start the parameter server process that listens to the public address 'server_ip:server:port' for clients. Each client can connect to this public address to join in training.
        >>> import flgo.algorithm.realbase as realbase
        >>> task = flgo.gen_task(...)
        >>> server_runner = flgo.init(task, realbase, scene='real_hserver')
        >>> server_runner.run(port='5555')
        Args:
            ip (str): ip address
            port (str): public port for client registration, default is 5555
            port_task (str): public port for client pulling task
            protocol (str): the communication protocol, default is TCP
        """
        if 'real' in self.option['scene']: self._read_zipped_task(ignore_names=no_zip_flags)
        self.logger = self.logger(task=self.option['task'], option=self.option, name=self.name+'_'+str(self.logger), level=self.option['log_level'])
        self.logger.register_variable(object=self, server=self)
        self._clients = {}
        self.ip = ip
        self.port = port
        self._lock_registration = threading.Lock()
        self._buffer = queue.Queue()
        self._lock_buffer = threading.Lock()
        self._exit = False
        self._lock_exit = threading.Lock()

        self.context = zmq.Context()
        self.registrar = self.context.socket(zmq.ROUTER)
        self.registrar.bind("%s://%s:%s" % (protocol, ip, port))

        self.port_task = self.get_free_port() if port_task == '' else port_task
        self.task_pusher = self.context.socket(zmq.ROUTER)
        self.task_pusher.bind("%s://%s:%s" % (protocol, ip, self.port_task))
        self.logger.info("Publish Task %s in %s://%s:%s"% (os.path.basename(self.option['task']),protocol, ip, self.port_task))

        self.port_send = self.get_free_port()
        self.sender = self.context.socket(zmq.PUB)
        self.sender.bind("%s://%s:%s" %(protocol, ip, self.port_send))

        self.port_recv = self.get_free_port()
        self.receiver = self.context.socket(zmq.PULL)
        self.receiver.bind("%s://%s:%s"%(protocol, ip, self.port_recv))

        self.port_alive = self.get_free_port()
        self.alive_detector = self.context.socket(zmq.ROUTER)
        self.alive_detector.bind("%s://%s:%s"%(protocol, ip, self.port_alive))
        self._lock_alive = threading.Lock()
        self._alive_state = {}

        self._poller = zmq.Poller()
        self._poller.register(self.task_pusher, zmq.POLLIN)
        self._poller.register(self.registrar, zmq.POLLIN)
        self._poller.register(self.receiver, zmq.POLLIN)
        self._poller.register(self.alive_detector, zmq.POLLIN)
        self._thread_listening = threading.Thread(target=self._listen)
        self._thread_listening.start()

        self.register()

        self.current_round = 1
        self.logger.time_start('Total Time Cost')
        if not self._load_checkpoint() and self.eval_interval>0:
            # evaluating initial model performance
            self.logger.info("--------------Initial Evaluation--------------")
            self.logger.time_start('Eval Time Cost')
            self.logger.log_once()
            self.logger.time_end('Eval Time Cost')
        while self.current_round<=self.num_rounds:
            updated = self.iterate()
            if updated is True or updated is None:
                self.logger.info("--------------Round {}--------------".format(self.current_round))
                if self.logger.check_if_log(self.current_round, self.eval_interval):
                    self.logger.time_start('Eval Time Cost')
                    self.logger.log_once()
                    self.logger.time_end('Eval Time Cost')
                    self._save_checkpoint()
                # check if early stopping
                if self.logger.early_stop(): break
                self.current_round += 1
        self.logger.info("=================End==================")
        self.logger.time_end('Total Time Cost')
        # save results as .json file
        self.logger.save_output_as_json()
        self.do_exit()
        self.communicate([_ for _ in range(len(self.clients))], mtype='close')
        exit(0)

    def aggregate(self, models: list, *args, **kwargs):
        if len(models)==0: return self.model
        return flgo.utils.fmodule._model_average(models).to(self.device)

    def unpack(self, pkgs:dict):
        if len(pkgs)==0: return collections.defaultdict(list)
        keys = list(list(pkgs.values())[0].keys())
        res = {}
        for k in keys:
            res[k] = []
            for cip in pkgs:
                v = pkgs[cip][k]
                res[k].append(v)
        return res

    def communicate_with(self, target_id, package={}):
        self.sender.send_string(target_id, zmq.SNDMORE)
        self.sender.send_pyobj(package)

    def communicate(self, selected_clients, mtype=0, asynchronous=False, only_available=True):
        avl_clients = self.available_clients
        avl_selected_clients = []
        uavl_selected_clients = []
        for i in selected_clients:
            if i in avl_clients.keys():
                avl_selected_clients.append(i)
            else:
                uavl_selected_clients.append(i)
        if only_available:
            self.selected_clients = avl_selected_clients
            for i in uavl_selected_clients:
                self.logger.info(f"Selected client {self.clients[i]} is dropped since it is currently not available.")
        else:
            for i in uavl_selected_clients:
                warnings.warn(f"Selected client {self.clients[i]} is dropped since it is currently not available.")
        selected_clients = [self.clients[i] for i in self.selected_clients]
        self.model.to('cpu')
        for i, name in enumerate(selected_clients):
            package = self.pack(i, mtype=mtype)
            package['__mtype__'] = mtype
            package['__round__'] = self.current_round
            package['name'] = name
            self.communicate_with(name, package)
        buffer = {}
        start_timestamp = time.time()
        while not self.is_exit():
            new_comings = self.clear_buffer()
            buffer.update(new_comings)
            crt_cost = time.time() - start_timestamp
            if asynchronous or all([(name in buffer) for name in selected_clients]) or crt_cost>self._communication_timeout:
                if crt_cost>self._communication_timeout:
                    timeout_clients = [name for name in selected_clients if name not in buffer.keys()]
                    for name in timeout_clients:
                        self.logger.info(f"Failed to receive packages from Client {name} due to timeout {self._communication_timeout}s. Using 'set_communication_timeout' method to set timeout can allow a longer waiting period.")
                break
            time.sleep(0.1)
        return self.unpack(buffer)

    def global_test(self, model=None, flag: str = 'val'):
        all_metrics = self.communicate([_ for _ in range(len(self.clients))], mtype='%s_metric'%flag)
        return all_metrics

    def get_free_port(self):
        import socket
        sock = socket.socket()
        sock.bind(('', 0))
        ip, port = sock.getsockname()
        sock.close()
        return port

    def init_algo_para(self, algo_para: dict):
        self.algo_para = algo_para
        if len(self.algo_para) == 0: return
        # initialize algorithm-dependent hyperparameters from the input options
        if self.option['algo_para'] is not None:
            # assert len(self.algo_para) == len(self.option['algo_para'])
            keys = list(self.algo_para.keys())
            for i, pv in enumerate(self.option['algo_para']):
                if i == len(self.option['algo_para']): break
                para_name = keys[i]
                try:
                    self.algo_para[para_name] = type(self.algo_para[para_name])(pv)
                except:
                    self.algo_para[para_name] = pv
        for para_name, value in self.algo_para.items():
            self.__setattr__(para_name, value)

    def save_checkpoint(self):
        cpt = {
            'round': self.current_round,
            'model_state_dict': self.model.state_dict(),
            'early_stop_option': {
                '_es_best_score': self.logger._es_best_score,
                '_es_best_round': self.logger._es_best_round,
                '_es_patience': self.logger._es_patience,
            },
            'output': self.logger.output,
        }
        return cpt

    def load_checkpoint(self, cpt):
        md = cpt.get('model_state_dict', None)
        round = cpt.get('round', None)
        output = cpt.get('output', None)
        early_stop_option = cpt.get('early_stop_option', None)
        time = cpt.get('time', None)
        learning_rate = cpt.get('learning_rate', None)
        if md is not None: self.model.load_state_dict(md)
        if round is not None: self.current_round = round + 1
        if output is not None: self.gv.logger.output = output
        if time is not None: self.gv.clock.set_time(time)
        if learning_rate is not None: self.learning_rate = learning_rate
        if early_stop_option is not None:
            self.gv.logger._es_best_score = early_stop_option['_es_best_score']
            self.gv.logger._es_best_round = early_stop_option['_es_best_round']
            self.gv.logger._es_patience = early_stop_option['_es_patience']

class Client(fedavg.Client):
    def __init__(self, option={}):
        super(Client, self).__init__(option)
        self.actions = {
            '0': self.reply,
        }
        self.timeout_register = 5
        self.heartbeat_interval = 30

    def val_metric(self, package):
        model = package['model']
        metrics = self.test(model, 'val')
        return metrics

    def test_metric(self, package):
        model = package['model']
        metrics = self.test(model, 'test')
        return metrics

    def train_metric(self, package):
        model = package['model']
        metrics = self.test(model, 'train')
        return metrics

    def set_timeout_register(self, t=5):
        self.timeout_register = t

    def register(self):
        self.logger.info("%s Registering..." % self.name)
        if self.timeout_register>0:
            self.registrar.setsockopt(zmq.RCVTIMEO, self.timeout_register*1000)
        try:
            self.registrar.send_pyobj({"name": self.name})
            reply = self.registrar.recv_pyobj()
        except zmq.error.Again as e:
            self.logger.info("Server temporarily unavailable")
            self.do_exit()
            exit(0)

        if '__option__' in reply: self.set_option(reply['__option__'])
        if 'algo_para' in reply and isinstance(reply['algo_para'], dict):
            self.option['algo_para'] = reply['algo_para']
            for k,v in reply['algo_para'].items(): setattr(self, k, v)
        return reply["port_recv"], reply["port_send"], reply['port_alive']

    def message_handler(self, package, *args, **kwargs):
        mtype = package['__mtype__']
        if package['__mtype__'] == 'close':
            self.sender.send_string(self.name, zmq.SNDMORE)
            self.sender.send_pyobj({'__mtype__': "close"})
            return True
        action = self.default_action if mtype not in self.actions else self.actions[mtype]
        response = action(package)
        assert isinstance(response, dict)
        response['__name__'] = self.name
        if hasattr(self, 'round'): response['__round__'] = self.round
        self.sender.send_string(self.name, zmq.SNDMORE)
        msg = pickle.dumps(response, pickle.DEFAULT_PROTOCOL)
        self.logger.info("{} Sending the package of size {}MB to the server...".format(self.name, len(msg)/1024/1024))
        self.sender.send(msg)
        # self.sender.send_pyobj(response)
        return False

    def is_exit(self):
        self._lock_exit.acquire()
        res = self._exit
        self._lock_exit.release()
        return res

    def do_exit(self):
        self._lock_exit.acquire()
        self._exit = True
        self._lock_exit.release()
        self.logger.info("%s was closed" % self.name)
        return

    def _heart_beat(self):
        while not self.is_exit():
            try:
                time.sleep(self.heartbeat_interval)
                self.heart_beator.send(b"")
            except Exception as e:
                self.logger.info(e)
                continue

    def _listen(self):
        while not self.is_exit():
            events = dict(self._poller.poll(10000))
            if self.heart_beator in events and events[self.heart_beator] == zmq.POLLIN:
                server_is_alive = self.heart_beator.recv()
            if self.receiver in events and events[self.receiver]==zmq.POLLIN:
                name = self.receiver.recv_string()
                assert name==self.name
                package_msg = self.receiver.recv()
                package_size = len(package_msg)
                package = self.receiver._deserialize(package_msg, pickle.loads)
                # package = self.receiver.recv_pyobj()
                assert '__mtype__' in package
                package['__size__'] = package_size/1024/1024 #MB
                if '__round__' in package.keys():
                    self.round = package['__round__']
                    self.logger.info("{} is selected at round {} and has received the package of {}MB".format(self.name, package['__round__'], package['__size__']))
                do_break = self.message_handler(package)
                if do_break: break
        return

    def run(self, server_ip:str='127.0.0.1', server_port: str='5555', protocol:str='tcp', timeout_register=10):
        """
        Start the client process that connects to the public address 'server_ip:server:port' of the parameter server..
        >>> import flgo.algorithm.realbase as realbase
        >>> import flgo
        >>> server_ip = '127.0.0.1'
        >>> server_task_port = ...
        >>> server_register_port = '5555'
        >>> task = ...
        >>> flgo.pull_task_from_("tcp://{}:{}".format(server_ip, server_task_port), task)
        >>> # set local dataset in task/dataset.py before joining in training
        >>> client_runner = flgo.init(task, realbase, scene='real_hclient')
        >>> client_runner.run(server_ip, server_register_port)
        Args:
            server_ip (str): ip address
            server_port (str): public port for client registration, default is 5555
            protocol (str): the communication protocol, default is TCP
            timeout_register (int): the timeout for registeration (seconds), defaulf is 10
        """
        self.logger = self.logger(task=self.option['task'], option=self.option, name=self.name+'_'+str(self.logger), level=self.option['log_level'])
        self.logger.register_variable(object=self, clients = [self])

        self.timeout_register = timeout_register
        self._exit = False
        self._lock_exit = threading.Lock()
        self.actions.update({'val_metric': self.val_metric, 'train_metric': self.train_metric, 'test_metric': self.test_metric,})

        self.context = zmq.Context()
        # Registration Socket
        self.registrar = self.context.socket(zmq.REQ)
        self.registrar.connect("%s://%s:%s"%(protocol, server_ip, server_port))
        port_svr_recv, port_svr_send, port_svr_alive = self.register()
        self.logger.info(f"{self.name} Successfully Registered to {server_ip}:{server_port}")

        self.receiver = self.context.socket(zmq.SUB)
        self.receiver.connect("%s://%s:%s" % (protocol, server_ip, port_svr_send))
        self.receiver.subscribe(self.name)

        self.heart_beator = self.context.socket(zmq.DEALER)
        self.heart_beator.setsockopt(zmq.IDENTITY, self.name.encode('utf-8'))
        self.heart_beator.connect("%s://%s:%s" % (protocol, server_ip, port_svr_alive))

        self.sender = self.context.socket(zmq.PUSH)
        self.sender.connect("%s://%s:%s" % (protocol, server_ip, port_svr_recv))

        self._poller = zmq.Poller()
        self._poller.register(self.receiver, zmq.POLLIN)
        self._poller.register(self.heart_beator, zmq.POLLIN)

        self._thread_heartbeat = threading.Thread(target=self._heart_beat)
        self._thread_heartbeat.start()

        self.logger.info(f"{self.name} Ready For Training...")
        self.logger.info("---------------------------------------------------------------------")
        # threading.Thread(target=self._heart_beat).start()
        self._server_timeout = np.inf
        self._server_alive_timestamp = time.time()
        while True:
            events = dict(self._poller.poll(10000))
            # if self.heart_beator in events and events[self.heart_beator] == zmq.POLLIN:
            #     _ = self.heart_beator.recv()
            #     self._server_alive_timestamp = time.time()
            if self.receiver in events and events[self.receiver]==zmq.POLLIN:
                name = self.receiver.recv_string()
                assert name==self.name
                package_msg = self.receiver.recv()
                package_size = len(package_msg)
                package = self.receiver._deserialize(package_msg, pickle.loads)
                # package = self.receiver.recv_pyobj()
                assert '__mtype__' in package
                if package['__mtype__']=='close':
                    self.sender.send_string(self.name, zmq.SNDMORE)
                    self.sender.send_pyobj({'__mtype__':"close"})
                    break
                package['__size__'] = package_size/1024/1024 #MB
                if '__round__' in package.keys():
                    self.round = package['__round__']
                    self.logger.info("{} is selected at round {} and has received the package of {}MB".format(self.name, package['__round__'], package['__size__']))
                self.message_handler(package)
            if time.time() - self._server_alive_timestamp>=self._server_timeout:
                self.logger.info("Lose connection to the server")
                self.do_exit()
                break
        torch.cuda.empty_cache()
        self.do_exit()
        exit(0)

    def set_option(self, option:dict={}):
        valid_keys = ['num_steps', 'learning_rate', 'batch_size', 'momentum', 'weight_decay', 'num_epochs', 'optimizer']
        types = [int, float, float, float, float, int, str]
        self.option.update(option)
        for k,t in zip(valid_keys, types):
            if k in option:
                try:
                    setattr(self, k, t(option[k]))
                except:
                    self.logger.info("Failed to set hyper-parameter {}={}".format(k, option[k]))
                    continue
        # correct hyper-parameters
        if hasattr(self, 'train_data') and self.train_data is not None:
            import math
            self.datavol = len(self.train_data)
            if hasattr(self, 'batch_size'):
                # reset batch_size
                if self.batch_size < 0:
                    self.batch_size = self.datavol
                elif self.batch_size >= 1:
                    self.batch_size = int(self.batch_size)
                else:
                    self.batch_size = int(self.datavol * self.batch_size)
            # reset num_steps
            if hasattr(self, 'num_steps') and hasattr(self, 'num_epochs'):
                if self.num_steps > 0:
                    self.num_epochs = 1.0 * self.num_steps / (math.ceil(self.datavol / self.batch_size))
                else:
                    self.num_steps = self.num_epochs * math.ceil(self.datavol / self.batch_size)
        return

    def set_data(self, data, flag:str='train') -> None:
        r"""
        Set self's attibute 'xxx_data' to be data where xxx is the flag. For example,
        after calling self.set_data([1,2,3], 'test'), self.test_data will be [1,2,3].
        Particularly, If the flag is 'train', the batchsize and the num_steps will be
        reset.

        Args:
            data: anything
            flag (str): the name of the data
        """
        setattr(self, flag + '_data', data)
        if flag not in self._data_names:
            self._data_names.append(flag)
        if flag == 'train':
            if data is None:
                warnings.warn("Local train data is None")
                return
            import math
            self.datavol = len(data)
            if hasattr(self, 'batch_size'):
                # reset batch_size
                if self.batch_size < 0:
                    self.batch_size = len(self.get_data(flag))
                elif self.batch_size >= 1:
                    self.batch_size = int(self.batch_size)
                else:
                    self.batch_size = int(self.datavol * self.batch_size)
            # reset num_steps
            if hasattr(self, 'num_steps') and hasattr(self, 'num_epochs'):
                if self.num_steps > 0:
                    self.num_epochs = 1.0 * self.num_steps / (math.ceil(self.datavol / self.batch_size))
                else:
                    self.num_steps = self.num_epochs * math.ceil(self.datavol / self.batch_size)

if __name__=='__main__':
    class algo:
        Server = Server
        Client = Client
    mlp.set_start_method('spawn', force=True)
    mlp.set_sharing_strategy('file_system')
    import flgo.benchmark.mnist_classification as mnist
    import flgo.benchmark.partition as fbp
    flgo.gen_task_by_(mnist, fbp.IIDPartitioner(num_clients=10), 'my_task')
    runner = flgo.init('my_task', algo, option={'proportion':0.2, 'gpu':[0], 'server_with_cpu':True, 'num_rounds':10, 'num_steps':1, 'log_file':True, 'log_level':'DEBUG'}, scene='parallel_horizontal')
    runner.run()