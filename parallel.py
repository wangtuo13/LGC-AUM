import os
import signal
from multiprocessing import Process, Manager

cmd_test = [
    'python helloworld.py --name tuo0',
    'python helloworld.py --name tuo1',
    'python helloworld.py --name tuo2',
    'python helloworld.py --name tuo3',
    'python helloworld.py --name tuo4',
    'python helloworld.py --name tuo5',
    'python helloworld.py --name tuo6',
    'python helloworld.py --name tuo7',
    'python helloworld.py --name tuo8',
    'python helloworld.py --name tuo9',
]

cmd_pretrain = [
    'python pretrain.py --name Palm --input_size 256 --n_cluster 100 --hidden_dim 100 --shuffle --pretrain \
    --pretrain_batchsize 256 --pretrain_epoch 300 --pretrain_lr 0.001 --noise 0.2\
    --pretrain_optim Adam --net VAE --channel 1 --wide 16 --num_workers 0 --plot',

    'python pretrain.py --name Mnist --input_size 784 --n_cluster 10 --hidden_dim 10 --shuffle --pretrain \
    --pretrain_batchsize 256 --pretrain_epoch 200 --pretrain_lr 0.001 --noise 0.2\
    --pretrain_optim Adam --net VAE --channel 1 --wide 28 --num_workers 0 --plot',

    'python pretrain.py --name Yale --input_size 1024 --n_cluster 38 --hidden_dim 38 --shuffle --pretrain \
    --pretrain_batchsize 256 --pretrain_epoch 200 --pretrain_lr 0.001 --noise 0.2\
    --pretrain_optim Adam --net VAE --channel 1 --wide 32 --num_workers 0 --plot',

    'python pretrain.py --name Fashion --input_size 784 --n_cluster 10 --hidden_dim 10 --shuffle --pretrain \
    --pretrain_batchsize 256 --pretrain_epoch 200 --pretrain_lr 0.001 --noise 0.2\
    --pretrain_optim Adam --net VAE --channel 1 --wide 28 --num_workers 0 --plot',

    'python pretrain.py --name USPS --input_size 256 --n_cluster 10 --hidden_dim 10 --shuffle --pretrain\
    --pretrain_batchsize 256 --pretrain_epoch 500 --pretrain_lr 0.0001 --noise 0.2\
    --pretrain_optim Adam --net VAE --channel 1 --wide 16 --num_workers 0 --plot',

    'python pretrain.py --name Coil_20 --input_size 16384 --n_cluster 20 --hidden_dim 20 --shuffle --pretrain \
    --pretrain_batchsize 256 --pretrain_epoch 300 --pretrain_lr 0.001 --noise 0.2\
    --pretrain_optim Adam --net VAE --channel 3 --wide 16 --num_workers 0 --plot',

    'python pretrain.py --name Cifar_10 --input_size 3072 --n_cluster 10 --hidden_dim 10 --shuffle --pretrain\
    --pretrain_batchsize 100 --pretrain_epoch 250 --pretrain_lr 0.0001 --noise 0.2\
    --pretrain_optim Adam --net VAE --channel 3 --wide 32 --num_workers 0 --plot',
]

cmd_two_stage = [
    'python two_stage.py --name Palm --input_size 256 --n_cluster 100 --hidden_dim 100 \
    --pretrain_batchsize 256 --pretrain_epoch 300 --pretrain_lr 0.001 --noise 0.2\
    --pretrain_optim Adam --net VAE --channel 1 --wide 16 --num_workers 0 --plot\
    --cluster_batchsize 1000 --cluster_epoch 500 --cluster_lr 0.01 --cluster_optim Adam\
    --manifold_lr 0.01 --change 1 --m 10 --divide 0.03 --total 2000',

    'python two_stage.py --name Mnist --input_size 784 --n_cluster 10 --hidden_dim 10\
    --pretrain_batchsize 256 --pretrain_epoch 200 --pretrain_lr 0.001 --noise 0.2\
    --pretrain_optim Adam --net VAE --channel 1 --wide 28 --num_workers 0 --plot\
    --cluster_batchsize 1000 --cluster_epoch 8000 --cluster_lr 0.001 --cluster_optim Adam\
     --manifold_lr 0.01 --change 2 --m 10 --divide 0 --total 70000 --Acc 80.16',

    'python two_stage.py --name Yale --input_size 1024 --n_cluster 38 --hidden_dim 38 \
    --pretrain_batchsize 256 --pretrain_epoch 300 --pretrain_lr 0.001 --noise 0.2\
    --pretrain_optim Adam --net VAE --channel 1 --wide 32 --num_workers 0 --plot\
    --cluster_batchsize 1000 --cluster_epoch 500 --cluster_lr 0.01 --cluster_optim Adam\
    --manifold_lr 0.01 --change 30 --m 10 --divide 0.01 --total 2000',

    'python two_stage.py --name Fashion --input_size 784 --n_cluster 10 --hidden_dim 10\
    --pretrain_batchsize 256 --pretrain_epoch 300 --pretrain_lr 0.001 --noise 0.2\
    --pretrain_optim Adam --net VAE --channel 1 --wide 28 --num_workers 0 --plot\
    --cluster_batchsize 1000 --cluster_epoch 8000 --cluster_lr 0.001 --cluster_optim Adam\
     --manifold_lr 0.001 --change 2 --m 10 --divide 0 --total 70000 --Acc 56.92'

    'python two_stage.py --name USPS --input_size 256 --n_cluster 10 --hidden_dim 10 \
    --pretrain_batchsize 256 --pretrain_epoch 300 --pretrain_lr 0.001 --noise 0.2\
    --pretrain_optim Adam --net VAE --channel 1 --wide 16 --num_workers 0 --plot\
    --cluster_batchsize 1000 --cluster_epoch 500 --cluster_lr 0.01 --cluster_optim Adam\
    --manifold_lr 0.01 --change 30 --m 10 --divide 0.03 --total 2000',

    'python two_stage.py --name Coil_20 --input_size 16384 --n_cluster 20 --hidden_dim 20 \
    --pretrain_batchsize 256 --pretrain_epoch 300 --pretrain_lr 0.001 --noise 0.2\
    --pretrain_optim Adam --net VAE --channel 3 --wide 16 --num_workers 0 --plot\
    --cluster_batchsize 1000 --cluster_epoch 500 --cluster_lr 0.01 --cluster_optim Adam\
    --manifold_lr 0.01 --change 30 --m 10 --divide 0.03 --total 2000',

    'python two_stage.py --name Cifar_10 --input_size 3072 --n_cluster 10 --hidden_dim 10\
    --pretrain_batchsize 100 --pretrain_epoch 250 --pretrain_lr 0.0001 --noise 0.2\
    --pretrain_optim Adam --net VAE --channel 3 --wide 32 --num_workers 0 --plot\
    --cluster_batchsize 1000 --cluster_epoch 500 --cluster_lr 0.001 --cluster_optim Adam\
    --manifold_lr 0.001 --change 30 --m 10 --divide 0.03 --total 2000',
]

cmd_run = [
    'python two_stage.py --name USPS --input_size 256 --n_cluster 10 --hidden_dim 10 \
    --pretrain_batchsize 256 --pretrain_epoch 300 --pretrain_lr 0.0001 --noise 0.2\
    --pretrain_optim Adam --net VAE --channel 1 --wide 16 --num_workers 0 --plot\
    --cluster_batchsize 1000 --cluster_epoch 1000 --cluster_lr 0.001 --cluster_optim Adam\
    --manifold_lr 0.001 --change 30 --m 10 --divide 0 --total 9298 --Acc 74.67',

    'python two_stage.py --name USPS --input_size 256 --n_cluster 10 --hidden_dim 10 \
    --pretrain_batchsize 256 --pretrain_epoch 300 --pretrain_lr 0.0001 --noise 0.2\
    --pretrain_optim Adam --net VAE --channel 1 --wide 16 --num_workers 0 --plot\
    --cluster_batchsize 1000 --cluster_epoch 1000 --cluster_lr 0.001 --cluster_optim Adam\
    --manifold_lr 0.001 --change 30 --m 20 --divide 0 --total 9298 --Acc 74.67',

    'python two_stage.py --name USPS --input_size 256 --n_cluster 10 --hidden_dim 10 \
    --pretrain_batchsize 256 --pretrain_epoch 300 --pretrain_lr 0.0001 --noise 0.2\
    --pretrain_optim Adam --net VAE --channel 1 --wide 16 --num_workers 0 --plot\
    --cluster_batchsize 1000 --cluster_epoch 1000 --cluster_lr 0.001 --cluster_optim Adam\
    --manifold_lr 0.001 --change 30 --m 5 --divide 0 --total 9298 --Acc 74.67',

    'python two_stage.py --name USPS --input_size 256 --n_cluster 10 --hidden_dim 10 \
    --pretrain_batchsize 256 --pretrain_epoch 300 --pretrain_lr 0.0001 --noise 0.2\
    --pretrain_optim Adam --net VAE --channel 1 --wide 16 --num_workers 0 --plot\
    --cluster_batchsize 1000 --cluster_epoch 1000 --cluster_lr 0.001 --cluster_optim Adam\
    --manifold_lr 0.001 --change 30 --m 15 --divide 0.1 --total 9298 --Acc 74.67',

    'python two_stage.py --name USPS --input_size 256 --n_cluster 10 --hidden_dim 10 \
    --pretrain_batchsize 256 --pretrain_epoch 300 --pretrain_lr 0.0001 --noise 0.2\
    --pretrain_optim Adam --net VAE --channel 1 --wide 16 --num_workers 0 --plot\
    --cluster_batchsize 1000 --cluster_epoch 1000 --cluster_lr 0.001 --cluster_optim Adam\
    --manifold_lr 0.001 --change 30 --m 15 --divide 0.5 --total 9298 --Acc 74.67',

]

cmd = cmd_run

def run(command, gpuid):
    os.system(command)

def term(sig_num, addtion):
    print('terminate process {}'.format(os.getpid()))
    try:
        print('the processes is {}'.format(processes))
        for p in processes:
            print('process {} terminate'.format(p.pid))
            p.terminate()
    except Exception as e:
        print(str(e))


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, term)

    processes = []
    idx = 0

    while idx < len(cmd):
        print(idx)
        p = Process(target=run, args=(cmd[idx], 0), name=str(0))
        p.start()

        processes.append(p)
        idx += 1

    for p in processes:
        p.join()
