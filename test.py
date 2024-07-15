import flgo.algorithm.fedavg as fedavg
import flgo.algorithm.fedfa as fedfa
import flgo.algorithm.afl as afl
import flgo.experiment.analyzer
import flgo.benchmark.partition as fbp
import os
import torch

task = './Ultrasound_4C'

def index_func(data):
    group = [-1 for _ in range(len(data))]
    # Attach each sample with a label of pixel density
    for i, id in enumerate(data.ids):
        if data.coco.imgs[id]['from'] == 1:
            group[i - 1] = 0
        if data.coco.imgs[id]['from'] == 2:
            group[i - 1] = 1
        if data.coco.imgs[id]['from'] == 3:
            group[i - 1] = 2
    return group

gen_config = {
    'benchmark':{'name':'flgo.benchmark.coco_detection_fetus'},
    'partitioner':{
        'name':fbp.IDPartitioner, # IDPartitioner根据每个样本所属的ID直接构造用户，每个用户对应一个ID
        'para':{
            'index_func': index_func
        }
    }
}

if __name__ == '__main__':
    # generate federated task if task doesn't exist
    if not os.path.exists(task): flgo.gen_task(gen_config, task_path=task)
    # running fedavg on the specified task
    runner = flgo.init(
        task, 
        fedavg, 
        {
            'gpu':[0,],
            'log_file':True, 
            'num_rounds': 100,
            'num_epochs': 1,
            'batch_size': 4,
            'learning_rate': 0.001,
            'test_batch_size': 1,
            'use_amp': True,
        }
    )

    torch.cuda.empty_cache()

    # 设置环境变量优化内存分配，防止内存碎片化
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    runner.run()
    # visualize the experimental result
    # flgo.experiment.analyzer.show(analysis_plan)