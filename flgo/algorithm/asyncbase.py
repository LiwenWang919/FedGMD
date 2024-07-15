from flgo.algorithm.fedbase import BasicServer
from flgo.algorithm.fedbase import BasicClient as Client
import numpy as np

class AsyncServer(BasicServer):
    def __init__(self, option={}):
        super(AsyncServer, self).__init__(option)
        self.concurrent_clients = set()
        self.buffered_clients = set()

    def sample(self):
        """
        Sample clients under the limitation of the maximum numder of concurrent clients.
        Returns:
            Selected clients.
        """
        all_clients = self.available_clients if 'available' in self.sample_option else [cid for cid in range(self.num_clients)]
        all_clients = list(set(all_clients).difference(self.buffered_clients))
        clients_per_round = self.clients_per_round - len(self.concurrent_clients)
        if clients_per_round<=0: return []
        clients_per_round = max(min(clients_per_round, len(all_clients)), 1)
        # full sampling with unlimited communication resources of the server
        if 'full' in self.sample_option:
            return all_clients
        # sample clients
        elif 'uniform' in self.sample_option:
            # original sample proposed by fedavg
            selected_clients = list(np.random.choice(all_clients, clients_per_round, replace=False)) if len(all_clients) > 0 else []
        elif 'md' in self.sample_option:
            # the default setting that is introduced by FedProx, where the clients are sampled with the probability in proportion to their local_movielens_recommendation data sizes
            local_data_vols = [self.clients[cid].datavol for cid in all_clients]
            total_data_vol = sum(local_data_vols)
            p = np.array(local_data_vols) / total_data_vol
            selected_clients = list(np.random.choice(all_clients, clients_per_round, replace=True, p=p)) if len(all_clients) > 0 else []
        return selected_clients

    def package_handler(self, received_packages:dict):
        """
        Handle packages received from clients and return whether the global model is updated in this function.

        Args:
            received_packages (dict): a dict consisting of uploaded contents from clients
        Returns:
            is_model_updated (bool): True if the global model is updated in this function.
        """
        if self.is_package_empty(received_packages): return False
        self.model = self.aggregate(received_packages['model'])
        return True

    def is_package_empty(self, received_packages:dict):
        """
        Check whether the package dict is empty

        Returns:
            is_empty (bool): True if the package dict is empty
        """
        return len(received_packages['__cid']) == 0

    def iterate(self):
        """
        The procedure of the server at each moment. Compared to synchronous methods, asynchronous servers perform iterations in a time-level view instead of a round-level view.

        Returns:
            is_model_updated (bool): True if the global model is updated at the current iteration
        """
        self.selected_clients = self.sample()
        self.concurrent_clients.update(set(self.selected_clients))
        if len(self.selected_clients) > 0: self.gv.logger.info('Select clients {} at time {}.'.format(self.selected_clients, self.gv.clock.current_time))
        self.model._round = self.current_round
        received_packages = self.communicate(self.selected_clients, asynchronous=True)
        self.concurrent_clients.difference_update(set(received_packages['__cid']))
        self.buffered_clients.update(set(received_packages['__cid']))
        if len(received_packages['__cid'])>0: self.gv.logger.info('Receive new models from clients {} at time {}'.format(received_packages['__cid'], self.gv.clock.current_time))
        is_model_updated = self.package_handler(received_packages)
        if is_model_updated: self.buffered_clients = set()
        return is_model_updated