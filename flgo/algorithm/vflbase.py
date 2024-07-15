from __future__ import annotations

import flgo.utils.fmodule
from .fedbase import BasicParty
import collections
import torch
import torch.multiprocessing as mp

class PassiveParty(BasicParty):
    r"""This is the implementation of the passive party in vertival FL.
    The passive party owns only a part of data features without label information.

    Args:
        option (dict): running-time option
    """
    def __init__(self, option:dict):
        super().__init__()
        self.option = option
        self.actions = {0: self.forward, 1:self.backward, 2:self.forward_test}
        self.id = None
        # create local_movielens_recommendation dataset
        self.data_loader = None
        # local_movielens_recommendation calculator
        self.device = self.gv.apply_for_device()
        self.calculator = self.gv.TaskCalculator(self.device, option['optimizer'])
        # hyper-parameters for training
        self.optimizer_name = option['optimizer']
        self.lr = option['learning_rate']
        self.momentum = option['momentum']
        self.weight_decay = option['weight_decay']
        self.batch_size = option['batch_size']
        self.num_steps = option['num_steps']
        self.num_epochs = option['num_epochs']
        self.model = None
        self.test_batch_size = option['test_batch_size']
        self.loader_num_workers = option['num_workers']
        self.current_steps = 0
        # system setting
        self._effective_num_steps = self.num_steps
        self._latency = 0

    def forward(self, package:dict={}):
        r"""
        Local forward to computing the activations on local_movielens_recommendation features

        Args:
            package (dict): the package from the active party that contains batch information and the type of data

        Returns:
            passive_package (dict): the package that contains the activation to be sent to the active party
        """
        batch_ids = package['batch']
        tmp = {'train': self.train_data, 'val': self.val_data, 'test':self.test_data}
        dataset = tmp[package['data_type']]
        # select samples in batch
        self.activation = self.local_module(dataset.get_batch_by_id(batch_ids)[0].to(self.device))
        return {'activation': self.activation.clone().detach()}

    def backward(self, package):
        r"""
        Local backward to computing the gradients on local_movielens_recommendation modules

        Args:
            package (dict): the package from the active party that contains the derivations
        """
        derivation = package['derivation']
        self.update_local_module(derivation, self.activation)
        return

    def update_local_module(self, derivation, activation):
        r"""
        Update local_movielens_recommendation modules according to the derivation and the activation

        Args:
            derivation (Any): the derivation from the active party
            activation (Any): the local_movielens_recommendation computed activation
        """
        optimizer = self.calculator.get_optimizer(self.local_module, self.lr)
        loss_surrogat = (derivation*activation).sum()
        loss_surrogat.backward()
        optimizer.step()
        return

    def forward_test(self, package):
        r"""
        Local forward to computing the activations on local_movielens_recommendation features for testing

        Args:
            package (dict): the package from the active party that contains batch information and the type of data

        Returns:
            passive_package (dict): the package that contains the activation to be sent to the active party
        """
        batch_ids = package['batch']
        tmp = {'train': self.train_data, 'val': self.val_data, 'test':self.test_data}
        dataset = tmp[package['data_type']]
        # select samples in batch
        self.activation = self.local_module(dataset.get_batch_by_id(batch_ids)[0].to(self.device))
        return {'activation': self.activation}

class ActiveParty(PassiveParty):
    r"""
    This is the implementation of the active party in vertival FL. The active party owns
    the data label information and may also own parts of data features. If a active party owns
    data features, it is also a passive party simultaneously.

    Args:
        option (dict): running-time option
    """
    def __init__(self, option):
        super().__init__(option)
        self.actions = {0: self.forward, 1: self.backward,2:self.forward_test}
        self.device = torch.device('cpu') if option['server_with_cpu'] else self.gv.apply_for_device()
        self.calculator = self.gv.TaskCalculator(self.device, optimizer_name = option['optimizer'])
        # basic configuration
        self.task = option['task']
        self.eval_interval = option['eval_interval']
        self.num_parallels = option['num_parallels']
        # hyper-parameters during training process
        self.num_rounds = option['num_rounds']
        self.proportion = option['proportion']
        self.batch_size = option['batch_size']
        self.decay_rate = option['learning_rate_decay']
        self.lr_scheduler_type = option['lr_scheduler']
        self.lr = option['learning_rate']
        self.sample_option = option['sample']
        self.aggregation_option = option['aggregate']
        # systemic option
        self.tolerance_for_latency = 999999
        self.sending_package_buffer = [None for _ in range(9999)]
        # algorithm-dependent parameters
        self.algo_para = {}
        self.current_round = 1
        # all options
        self.option = option
        self.id = 0

    def communicate(self, selected_clients, mtype=0, asynchronous=False):
        """
        The whole simulating communication procedure with the selected clients.
        This part supports for simulating the client dropping out.
        Args:
            selected_clients: the clients to communicate with
        Returns:
            :the unpacked response from clients that is created ny self.unpack()
        """
        packages_received_from_clients = []
        received_package_buffer = {}
        communicate_clients = list(set(selected_clients))
        # prepare packages for clients
        for cid in communicate_clients:
            received_package_buffer[cid] = None
        # communicate with selected clients
        if self.num_parallels <= 1:
            # computing iteratively
            for client_id in communicate_clients:
                server_pkg = self.pack(client_id, mtype=mtype)
                server_pkg['__mtype__'] = mtype
                response_from_client_id = self.communicate_with(client_id, package=server_pkg)
                packages_received_from_clients.append(response_from_client_id)
        else:
            # computing in parallel with torch.multiprocessing
            pool = mp.Pool(self.num_parallels)
            for client_id in communicate_clients:
                server_pkg = self.pack(client_id, mtype=mtype)
                server_pkg['__mtype__'] = mtype
                self.clients[client_id].update_device(self.gv.apply_for_device())
                args = (int(client_id), server_pkg)
                packages_received_from_clients.append(pool.apply_async(self.communicate_with, args=args))
            pool.close()
            pool.join()
            packages_received_from_clients = list(map(lambda x: x.get(), packages_received_from_clients))
        for i,cid in enumerate(communicate_clients): received_package_buffer[cid] = packages_received_from_clients[i]
        packages_received_from_clients = [received_package_buffer[cid] for cid in selected_clients if received_package_buffer[cid]]
        self.received_clients = selected_clients
        return self.unpack(packages_received_from_clients)

    def unpack(self, packages_received_from_clients):
        """
        Unpack the information from the received packages. Return models and losses as default.
        Args:
            packages_received_from_clients (list of dict):
        Returns:
            res (dict): collections.defaultdict that contains several lists of the clients' reply
        """
        if len(packages_received_from_clients)==0: return collections.defaultdict(list)
        res = {pname:[] for pname in packages_received_from_clients[0]}
        for cpkg in packages_received_from_clients:
            for pname, pval in cpkg.items():
                res[pname].append(pval)
        return res

    def run(self):
        """
        Start the federated learning symtem where the global model is trained iteratively.
        """
        self.gv.logger.time_start('Total Time Cost')
        self.gv.logger.info("--------------Initial Evaluation--------------")
        self.gv.logger.time_start('Eval Time Cost')
        self.gv.logger.log_once()
        self.gv.logger.time_end('Eval Time Cost')
        while self.current_round <= self.num_rounds:
            # iterate
            updated = self.iterate()
            # using logger to evaluate the model if the model is updated
            if updated is True or updated is None:
                self.gv.logger.info("--------------Round {}--------------".format(self.current_round))
                # check log interval
                if self.gv.logger.check_if_log(self.current_round, self.eval_interval):
                    self.gv.logger.time_start('Eval Time Cost')
                    self.gv.logger.log_once()
                    self.gv.logger.time_end('Eval Time Cost')
                if self.gv.logger.early_stop(): break
                self.current_round += 1
        self.gv.logger.info("=================End==================")
        self.gv.logger.time_end('Total Time Cost')
        # save results as .json file
        self.gv.logger.save_output_as_json()
        return

    def iterate(self):
        r"""
        The standard VFL process.

         1. The active party first generates the batch information.

         2. Then, it collects activations from all the passive parties.

         3. Thirdly, it continues the forward passing and backward passing to update the decoder part of the model, and distributes the derivations to parties.

         4. Finally, each passive party will update its local_movielens_recommendation modules accoring to the derivations and activations.

        Returns:
            updated (bool): whether the model is updated in this iteration
        """
        self._data_type='train'
        self.crt_batch = self.get_batch_data()
        activations = self.communicate([p.id for p in self.parties], mtype=0)['activation']
        self.defusions = self.update_global_module(activations, self.global_module)
        _ = self.communicate([pid for pid in range(len(self.parties))], mtype=1)
        return True

    def pack(self, party_id, mtype=0):
        r"""
        Pack the necessary information to parties into packages.

        Args:
            party_id (int): the id of the party
            mtype (Any): the message type

        Returns:
            package (dict): the package
        """
        if mtype==0:
            return {'batch': self.crt_batch[2], 'data_type': self._data_type}
        elif mtype==1:
            return {'derivation': self.defusion[party_id]}
        elif mtype==2:
            return {'batch': self.crt_test_batch[2], 'data_type': self._data_type}

    def get_batch_data(self):
        """
        Get the batch of data
        Returns:
            batch_data (Any): a batch of data
        """
        try:
            batch_data = next(self.data_loader)
        except:
            self.data_loader = iter(self.calculator.get_dataloader(self.train_data, batch_size=self.batch_size))
            batch_data = next(self.data_loader)
        return batch_data

    def update_global_module(self, activations:list, model:torch.nn.Module|flgo.utils.fmodule.FModule):
        r"""
        Update the global module by computing the forward passing and the backward passing. The attribute
        self.defusion and self.fusion.grad will be changed after calling this method.

        Args:
            activations (list): a list of activations from all the passive parties
            model (torch.nn.Module|flgo.utils.fmodule.FModule): the model
        """
        self.fusion = self.fuse(activations)
        self.fusion.requires_grad=True
        optimizer = self.calculator.get_optimizer(self.global_module, lr=self.lr)
        loss = self.calculator.compute_loss(model, (self.fusion, self.crt_batch[1]))['loss']
        loss.backward()
        optimizer.step()
        self.defusion = self.defuse(self.fusion)

    def fuse(self, activations:list):
        r"""
        Fuse the activations into one.

        Args:
            activations (list): a list of activations from all the passive parties

        Returns:
            fusion (Any): the fused result
        """
        return torch.stack(activations).mean(dim=0)

    def defuse(self, fusion):
        r"""
        Defuse the fusion into derivations.

        Args:
            fusion (Any): the fused result

        Returns:
            derivations (list): a list of derivations
        """
        return [fusion.grad for _ in self.parties]

    def test(self, flag:str='test') -> dict:
        r"""
        Test the performance of the model

        Args:
            flag (str): the type of dataset

        Returns:
            result (dict): a dict that contains the testing result
        """
        self.set_model_mode('eval')
        flag_dict = {'test':self.test_data, 'train':self.train_data, 'val':self.val_data}
        dataset = flag_dict[flag]
        self._data_type = flag
        dataloader = self.calculator.get_dataloader(dataset, batch_size=128)
        total_loss = 0.0
        num_correct = 0
        for batch_id, batch_data in enumerate(dataloader):
            self.crt_test_batch = batch_data
            activations = self.communicate([pid for pid in range(len(self.parties))], mtype=2)['activation']
            fusion = self.fuse(activations)
            outputs = self.global_module(fusion.to(self.device))
            batch_mean_loss = self.calculator.criterion(outputs, batch_data[1].to(self.device)).item()
            y_pred = outputs.data.max(1, keepdim=True)[1].cpu()
            correct = y_pred.eq(batch_data[1].data.view_as(y_pred)).long().cpu().sum()
            num_correct += correct.item()
            total_loss += batch_mean_loss * len(batch_data[1])
        self.set_model_mode('train')
        return {'accuracy': 1.0 * num_correct / len(dataset), 'loss': total_loss / len(dataset)}

    def set_model_mode(self,mode = 'train'):
        r"""
        Set all the modes of the modules owned by all the parties.

        Args:
            mode (str): the mode of models
        """
        for party in self.parties:
            if hasattr(party, 'local_module') and party.local_module is not None:
                if mode == 'train':
                    party.local_module.train()
                else:
                    party.local_module.eval()
            if hasattr(party, 'global_module') and party.global_module is not None:
                if mode == 'train':
                    party.global_module.train()
                else:
                    party.global_module.eval()

    def init_algo_para(self, algo_para: dict):
        """
        Initialize the algorithm-dependent hyper-parameters for the server and all the clients.
        :param
            algo_paras (dict): the dict that defines the hyper-parameters (i.e. name, value and type) for the algorithm.

        Example 1:
            calling `self.init_algo_para({'u':0.1})` will set the attributions `server.u` and `c.u` as 0.1 with type float where `c` is an instance of `CLient`.
        Note:
            Once `option['algo_para']` is not `None`, the value of the pre-defined hyperparameters will be replaced by the list of values in `option['algo_para']`,
            which requires the length of `option['algo_para']` is equal to the length of `algo_paras`
        """
        self.algo_para = algo_para
        if len(self.algo_para)==0: return
        # initialize algorithm-dependent hyperparameters from the input options
        if self.option['algo_para'] is not None:
            # assert len(self.algo_para) == len(self.option['algo_para'])
            keys = list(self.algo_para.keys())
            for i,pv in enumerate(self.option['algo_para']):
                if i==len(self.option['algo_para']): break
                para_name = keys[i]
                try:
                    self.algo_para[para_name] = type(self.algo_para[para_name])(pv)
                except:
                    self.algo_para[para_name] = pv
        # register the algorithm-dependent hyperparameters as the attributes of the server and all the clients
        for para_name, value in self.algo_para.items():
            self.__setattr__(para_name, value)
            for c in self.parties:
                if c.id!=self.id:
                    c.__setattr__(para_name, value)
        return