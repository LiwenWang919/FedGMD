## Algorithm Integration
We have already implemented 10+ SOTA algorithms in recent years' top tiers conferences and tiers.

| Method   |Reference| Publication    | Tag                                      |
|----------|---|----------------|------------------------------------------|
| FedAvg   |<a href='#refer-anchor-1'>[McMahan et al., 2017]</a>| AISTATS' 2017  ||
| FedAsync |<a href='#refer-anchor-2'>[Cong Xie et al., 2019]</a>|| Asynchronous   |
| FedBuff  |<a href='#refer-anchor-3'>[John Nguyen et al., 2022]</a>| AISTATS 2022   | Asynchronous                             |
| TiFL     |<a href='#refer-anchor-4'>[Zheng Chai et al., 2020]</a>| HPDC 2020      | Communication-efficiency, responsiveness |
| AFL      |<a href='#refer-anchor-5'>[Mehryar Mohri et al., 2019]</a>| ICML 2019      | Fairness                                 |
| FedFv     |<a href='#refer-anchor-6'>[Zheng Wang et al., 2019]</a>| IJCAI 2021     | Fairness                                 |
| FedMgda+     |<a href='#refer-anchor-7'>[Zeou Hu et al., 2022]</a>| IEEE TNSE 2022 | Fairness, robustness                     |
| FedProx     |<a href='#refer-anchor-8'>[Tian Li et al., 2020]</a>| MLSys 2020     | Non-I.I.D., Incomplete Training          |
| Mifa     |<a href='#refer-anchor-9'>[Xinran Gu et al., 2021]</a>| NeurIPS 2021   | Client Availability                      |
| PowerofChoice     |<a href='#refer-anchor-10'>[Yae Jee Cho et al., 2020]</a>| arxiv | Biased Sampling, Fast-Convergence  |
| QFedAvg     |<a href='#refer-anchor-11'>[Tian Li et al., 2020]</a>| ICLR 2020      | Communication-efficient,fairness         |
| Scaffold     |<a href='#refer-anchor-12'>[Sai Praneeth Karimireddy et al., 2020]</a>| ICML 2020      | Non-I.I.D., Communication Capacity       |


## Benchmark Gallary
| Benchmark   |Type| Scene    | Task                                     |
|----------|---|----------------|------------------------------------------|
| CIFAR100 | image| horizontal | classification || 
| CIFAR10  | image| horizontal | classification || 
| CiteSeer | graph | horizontal | classification || 
| Cora  |  graph | horizontal | classification || 
| PubMed  | graph | horizontal | classification || 
| MNIST  | image| horizontal | classification || 
| EMNIST  | image| horizontal | classification ||
| FEMINIST | image| horizontal | classification ||  
| FashionMINIST  | image| horizontal | classification || 
| ENZYMES  | graph| horizontal | classification || 
| Reddit  | text | horizontal | classification || 
| Sentiment140  | text| horizontal | classification || 
| MUTAG  | graph | horizontal | classification || 
| Shakespeare  | text | horizontal | classification || 
| Synthetic  | table| horizontal | classification || 

## Async/Sync Supported
We set a virtual global clock and a client-state machine to simulate a real-world scenario for comparison on asynchronous
 and synchronous strategies. Here we provide a comprehensive example to help understand the difference 
between the two strategies in FLGo.

![async_sync](https://raw.githubusercontent.com/WwZzz/myfigs/master/overview_flgo_async.png)
For synchronous algorithms, the server would wait for the slowest clients. 
In round 1,the server select a subset of idle clients (i.e. client i,u,v) 
to join in training and the slowest client v dominates the duration of this 
round (i.e. four time units). If there is anyone suffering from 
training failure (i.e. being dropped out), the duration of the current round 
should be the longest time that the server will wait for it (e.g. round 2 takes 
the maximum waiting time of six units to wait for response from client v). 

For asynchronous algorithms, the server usually periodically samples the idle 
clients to update models, where the length of the period is set as two time 
units in our example. After sampling the currently idle clients, the server will 
immediately checks whether there are packages currently returned from clients 
(e.g. the server selects client j and receives the package from client k at time 13). 

## Experimental Tools
For experimental purposes 

## Automatical Tuning

## Multi-Scene (Horizontal and Vertical)

## Accelerating by Multi-Process


## References
<div id='refer-anchor-1'></div>

\[McMahan. et al., 2017\] [Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-Efficient Learning of Deep Networks from Decentralized Data. In International Conference on Artificial Intelligence and Statistics (AISTATS), 2017.](https://arxiv.org/abs/1602.05629)

<div id='refer-anchor-2'></div>

\[Cong Xie. et al., 2019\] [Cong Xie, Sanmi Koyejo, Indranil Gupta. Asynchronous Federated Optimization. ](https://arxiv.org/abs/1903.03934)

<div id='refer-anchor-3'></div>

\[John Nguyen. et al., 2022\] [John Nguyen, Kshitiz Malik, Hongyuan Zhan, Ashkan Yousefpour, Michael Rabbat, Mani Malek, Dzmitry Huba. Federated Learning with Buffered Asynchronous Aggregation. In International Conference on Artificial Intelligence and Statistics (AISTATS), 2022.](https://arxiv.org/abs/2106.06639)

<div id='refer-anchor-4'></div>

\[Zheng Chai. et al., 2020\] [Zheng Chai, Ahsan Ali, Syed Zawad, Stacey Truex, Ali Anwar, Nathalie Baracaldo, Yi Zhou, Heiko Ludwig, Feng Yan, Yue Cheng. TiFL: A Tier-based Federated Learning System.In International Symposium on High-Performance Parallel and Distributed Computing(HPDC), 2020](https://arxiv.org/abs/2106.06639)

<div id='refer-anchor-5'></div>

\[Mehryar Mohri. et al., 2019\] [Mehryar Mohri, Gary Sivek, Ananda Theertha Suresh. Agnostic Federated Learning.In International Conference on Machine Learning(ICML), 2019](https://arxiv.org/abs/1902.00146)

<div id='refer-anchor-6'></div>

\[Zheng Wang. et al., 2021\] [Zheng Wang, Xiaoliang Fan, Jianzhong Qi, Chenglu Wen, Cheng Wang, Rongshan Yu. Federated Learning with Fair Averaging. In International Joint Conference on Artificial Intelligence, 2021](https://arxiv.org/abs/2104.14937#)

<div id='refer-anchor-7'></div>

\[Zeou Hu. et al., 2022\] [Zeou Hu, Kiarash Shaloudegi, Guojun Zhang, Yaoliang Yu. Federated Learning Meets Multi-objective Optimization. In IEEE Transactions on Network Science and Engineering, 2022](https://arxiv.org/abs/2006.11489)

<div id='refer-anchor-8'></div>

\[Tian Li. et al., 2020\] [Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, Virginia Smith. Federated Optimization in Heterogeneous Networks. In Conference on Machine Learning and Systems, 2020](https://arxiv.org/abs/1812.06127)

<div id='refer-anchor-9'></div>

\[Xinran Gu. et al., 2021\] [Xinran Gu, Kaixuan Huang, Jingzhao Zhang, Longbo Huang. Fast Federated Learning in the Presence of Arbitrary Device Unavailability. In Neural Information Processing Systems(NeurIPS), 2021](https://arxiv.org/abs/2106.04159)

<div id='refer-anchor-10'></div>

\[Yae Jee Cho. et al., 2020\] [Yae Jee Cho, Jianyu Wang, Gauri Joshi. Client Selection in Federated Learning: Convergence Analysis and Power-of-Choice Selection Strategies. ](https://arxiv.org/abs/2010.01243)

<div id='refer-anchor-11'></div>

\[Tian Li. et al., 2020\] [Tian Li, Maziar Sanjabi, Ahmad Beirami, Virginia Smith. Fair Resource Allocation in Federated Learning. In International Conference on Learning Representations, 2020](https://arxiv.org/abs/1905.10497)

<div id='refer-anchor-12'></div>