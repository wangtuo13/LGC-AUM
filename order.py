import os
import sys
sys.path.append(r'\home\ubuntu\workspace')

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
    --pretrain_batchsize 256 --pretrain_epoch 300 --pretrain_lr 0.001 --noise 0.2 --trans\
    --pretrain_optim Adam --net VAE --channel 1 --wide 16 --num_workers 0 --plot',

    'python pretrain.py --name Mnist --input_size 784 --n_cluster 10 --hidden_dim 10 --shuffle --pretrain \
    --pretrain_batchsize 256 --pretrain_epoch 200 --pretrain_lr 0.001 --noise 0.2 --trans\
    --pretrain_optim Adam --net VAE --channel 1 --wide 28 --num_workers 0 --plot',

    'python pretrain.py --name Mnist --input_size 784 --n_cluster 10 --hidden_dim 10 --shuffle --pretrain \
    --pretrain_batchsize 256 --pretrain_epoch 30 --pretrain_lr 0.001 --noise 0.2 --trans\
    --pretrain_optim Adam --net CNN_VAE2 --channel 1 --wide 28 --num_workers 0 --plot',

    'python pretrain.py --name Yale --input_size 1024 --n_cluster 38 --hidden_dim 38 --shuffle --pretrain \
    --pretrain_batchsize 256 --pretrain_epoch 200 --pretrain_lr 0.001 --noise 0.2 --trans\
    --pretrain_optim Adam --net VAE --channel 1 --wide 32 --num_workers 0 --plot',

    'python pretrain.py --name Fashion --input_size 784 --n_cluster 10 --hidden_dim 10 --shuffle --pretrain \
    --pretrain_batchsize 256 --pretrain_epoch 350 --pretrain_lr 0.0001 --noise 0.2 --trans\
    --pretrain_optim Adam --net VAE --channel 1 --wide 28 --num_workers 0 --plot',

    'python pretrain.py --name Fashion --input_size 784 --n_cluster 10 --hidden_dim 10 --shuffle --pretrain \
    --pretrain_batchsize 256 --pretrain_epoch 150 --pretrain_lr 0.001 --noise 0.2 --trans\
    --pretrain_optim Adam --net CNN_VAE2 --channel 1 --wide 28 --num_workers 0 --plot',

    'python pretrain.py --name Fashion2 --input_size 784 --n_cluster 10 --hidden_dim 10 --shuffle --pretrain \
    --pretrain_batchsize 256 --pretrain_epoch 200 --pretrain_lr 0.001 --noise 0.3 --trans\
    --pretrain_optim Adam --net VAE --channel 1 --wide 28 --num_workers 0 --plot',

    'python pretrain.py --name USPS --input_size 256 --n_cluster 10 --hidden_dim 10 --shuffle --pretrain \
    --pretrain_batchsize 256 --pretrain_epoch 300 --pretrain_lr 0.001 --noise 0.2 --trans\
    --pretrain_optim Adam --net VAE --channel 1 --wide 16 --num_workers 0 --plot',

    'python pretrain.py --name USPS --input_size 256 --n_cluster 10 --hidden_dim 10 --shuffle --pretrain \
    --pretrain_batchsize 256 --pretrain_epoch 150 --pretrain_lr 0.001 --noise 0.2 --trans\
    --pretrain_optim Adam --net CNN_VAE2 --channel 1 --wide 16 --num_workers 0 --plot',

    'python pretrain.py --name USPS2 --input_size 256 --n_cluster 10 --hidden_dim 10 --shuffle --pretrain \
    --pretrain_batchsize 256 --pretrain_epoch 300 --pretrain_lr 0.001 --noise 0.2 --trans\
    --pretrain_optim Adam --net VAE --channel 1 --wide 16 --num_workers 0 --plot',

    'python pretrain.py --name Coil_20 --input_size 16384 --n_cluster 20 --hidden_dim 20 --shuffle --pretrain \
    --pretrain_batchsize 256 --pretrain_epoch 300 --pretrain_lr 0.001 --noise 0.2 --trans\
    --pretrain_optim Adam --net VAE --channel 3 --wide 16 --num_workers 0 --plot',


    'python pretrain.py --name Cifar_10 --input_size 3072 --n_cluster 10 --hidden_dim 10 --shuffle --pretrain \
    --pretrain_batchsize 256 --pretrain_epoch 300 --pretrain_lr 0.001 --noise 0.2 --trans\
    --pretrain_optim Adam --net VAE --channel 3 --wide 32 --num_workers 0 --plot',

    'python pretrain.py --name STL_10 --input_size 27648 --n_cluster 10 --hidden_dim 10 --shuffle --pretrain \
    --pretrain_batchsize 256 --pretrain_epoch 400 --pretrain_lr 0.0001 --noise 0.2 --trans\
    --pretrain_optim Adam --net VAE --channel 3 --wide 96 --num_workers 0 --plot',

    'python pretrain.py --name STL_10 --input_size 27648 --n_cluster 10 --hidden_dim 10 --shuffle --pretrain \
    --pretrain_batchsize 256 --pretrain_epoch 300 --pretrain_lr 0.001 --noise 0.2 --trans\
    --pretrain_optim Adam --net VAE --channel 3 --wide 96 --num_workers 0 --plot',

    'python pretrain.py --name STL_10 --input_size 12288 --n_cluster 10 --hidden_dim 10 --shuffle --pretrain \
    --pretrain_batchsize 256 --pretrain_epoch 150 --pretrain_lr 0.001 --noise 0.2 --trans\
    --pretrain_optim Adam --net CNN_VAE2 --channel 3 --wide 64 --num_workers 0 --plot',

    'python pretrain.py --name tiny --input_size 12288 --n_cluster 200 --hidden_dim 200 --shuffle --pretrain \
    --pretrain_batchsize 256 --pretrain_epoch 150 --pretrain_lr 0.001 --noise 0.2 --trans\
    --pretrain_optim Adam --net CNN_VAE2 --channel 3 --wide 64 --num_workers 0 --plot',
]

cmd_two_stage = [
    'python two_stage.py --name Palm --input_size 256 --n_cluster 100 --hidden_dim 100 \
    --pretrain_batchsize 256 --pretrain_epoch 300 --pretrain_lr 0.001 --noise 0.2\
    --pretrain_optim Adam --net VAE --channel 1 --wide 16 --num_workers 0\
    --cluster_batchsize 1000 --cluster_epoch 500 --cluster_lr 0.01 --cluster_optim Adam\
    --manifold_lr 0.01 --change 1 --m 10 --divide 0.03 --total 2000',

    'python two_stage.py --name Palm --input_size 256 --n_cluster 100 --hidden_dim 100 \
    --pretrain_batchsize 256 --pretrain_epoch 150 --pretrain_lr 0.001 --noise 0.2\
    --pretrain_optim Adam --net CNN_VAE2 --channel 1 --wide 16 --num_workers 0\
    --cluster_batchsize 1000 --cluster_epoch 500 --cluster_lr 0.001 --cluster_optim Adam\
    --manifold_lr 0.001 --change 1 --m 5 --divide 0.1 --total 2000 --Acc 77.0',

    'python two_stage.py --name Mnist --input_size 784 --n_cluster 10 --hidden_dim 10\
    --pretrain_batchsize 256 --pretrain_epoch 350 --pretrain_lr 0.001 --noise 0.2\
    --pretrain_optim Adam --net VAE --channel 1 --wide 28 --num_workers 0\
    --cluster_batchsize 1000 --cluster_epoch 500 --cluster_lr 0.001 --cluster_optim Adam\
     --manifold_lr 0.001 --change 2 --m 10 --divide 0.1 --total 70000 --Acc 80.22',

    'python two_stage.py --name Mnist --input_size 784 --n_cluster 10 --hidden_dim 10\
    --pretrain_batchsize 256 --pretrain_epoch 30 --pretrain_lr 0.001 --noise 0.2\
    --pretrain_optim Adam --net CNN_VAE2 --channel 1 --wide 28 --num_workers 0\
    --cluster_batchsize 1000 --cluster_epoch 500 --cluster_lr 0.001 --cluster_optim Adam\
     --manifold_lr 0.0001 --change 2 --m 10 --divide 0.1 --total 70000 --Acc 72.69',

    'python two_stage.py --name Yale --input_size 1024 --n_cluster 38 --hidden_dim 38 \
    --pretrain_batchsize 256 --pretrain_epoch 300 --pretrain_lr 0.001 --noise 0.2\
    --pretrain_optim Adam --net VAE --channel 1 --wide 32 --num_workers 0\
    --cluster_batchsize 1000 --cluster_epoch 500 --cluster_lr 0.01 --cluster_optim Adam\
    --manifold_lr 0.01 --change 30 --m 10 --divide 0.01 --total 2000',

    'python two_stage.py --name Fashion --input_size 784 --n_cluster 10 --hidden_dim 10\
    --pretrain_batchsize 256 --pretrain_epoch 300 --pretrain_lr 0.001 --noise 0.2\
    --pretrain_optim Adam --net VAE --channel 1 --wide 28 --num_workers 0\
    --cluster_batchsize 1000 --cluster_epoch 8000 --cluster_lr 0.001 --cluster_optim Adam\
     --manifold_lr 0.001 --change 2 --m 10 --divide 0 --total 70000 --Acc 56.92'
    
    'python two_stage.py --name Fashion2 --input_size 784 --n_cluster 10 --hidden_dim 10\
    --pretrain_batchsize 256 --pretrain_epoch 300 --pretrain_lr 0.001 --noise 0.2\
    --pretrain_optim Adam --net VAE --channel 1 --wide 28 --num_workers 0\
    --cluster_batchsize 1000 --cluster_epoch 1000 --cluster_lr 0.001 --cluster_optim Adam\
    --manifold_lr 0.001 --change 2 --m 10 --divide 0.15 --total 70000 --Acc 55.35'
    
    'python two_stage.py --name Fashion2 --input_size 784 --n_cluster 10 --hidden_dim 10\
    --pretrain_batchsize 256 --pretrain_epoch 200 --pretrain_lr 0.001 --noise 0.1\
    --pretrain_optim Adam --net VAE --channel 1 --wide 28 --num_workers 0\
    --cluster_batchsize 1000 --cluster_epoch 300 --cluster_lr 0.001 --cluster_optim Adam\
    --manifold_lr 0.001 --change 2 --m 10 --divide 0.15 --total 70000 --Acc 53.71 --times 31'
    
    'python two_stage.py --name Fashion2 --input_size 784 --n_cluster 10 --hidden_dim 10\
    --pretrain_batchsize 256 --pretrain_epoch 200 --pretrain_lr 0.001 --noise 0.2\
    --pretrain_optim Adam --net VAE --channel 1 --wide 28 --num_workers 0\
    --cluster_batchsize 1000 --cluster_epoch 300 --cluster_lr 0.001 --cluster_optim Adam\
    --manifold_lr 0.001 --change 2 --m 10 --divide 0 --total 70000 --Acc 59.22 --times 2001',

    'python two_stage.py --name USPS --input_size 256 --n_cluster 10 --hidden_dim 10 \
    --pretrain_batchsize 256 --pretrain_epoch 300 --pretrain_lr 0.001 --noise 0.2\
    --pretrain_optim Adam --net VAE --channel 1 --wide 16 --num_workers 0\
    --cluster_batchsize 1000 --cluster_epoch 500 --cluster_lr 0.01 --cluster_optim Adam\
    --manifold_lr 0.01 --change 30 --m 10 --divide 0.03 --total 2000',

    'python two_stage.py --name USPS2 --input_size 256 --n_cluster 10 --hidden_dim 10 \
    --pretrain_batchsize 256 --pretrain_epoch 200 --pretrain_lr 0.001 --noise 0.1\
    --pretrain_optim Adam --net VAE --channel 1 --wide 16 --num_workers 0\
    --cluster_batchsize 1000 --cluster_epoch 400 --cluster_lr 0.001 --cluster_optim Adam\
    --manifold_lr 0.001 --change 30 --m 5 --divide 0.1 --total 9298 --Acc 73.33 --times 32',

    'python two_stage.py --name USPS2 --input_size 256 --n_cluster 10 --hidden_dim 10 \
    --pretrain_batchsize 256 --pretrain_epoch 60 --pretrain_lr 0.001 --noise 0.2\
    --pretrain_optim Adam --net CNN_VAE2 --channel 1 --wide 16 --num_workers 0\
    --cluster_batchsize 1000 --cluster_epoch 500 --cluster_lr 0.001 --cluster_optim Adam\
    --manifold_lr 0.001 --change 30 --m 10 --divide 0.1 --total 9298 --Acc 68.06',

    'python two_stage.py --name Coil_20 --input_size 16384 --n_cluster 20 --hidden_dim 20 \
    --pretrain_batchsize 256 --pretrain_epoch 300 --pretrain_lr 0.001 --noise 0.2\
    --pretrain_optim Adam --net VAE --channel 3 --wide 16 --num_workers 0\
    --cluster_batchsize 1000 --cluster_epoch 500 --cluster_lr 0.01 --cluster_optim Adam\
    --manifold_lr 0.01 --change 30 --m 10 --divide 0.03 --total 2000',

    'python two_stage.py --name Cifar_10 --input_size 3072 --n_cluster 10 --hidden_dim 10 \
    --pretrain_batchsize 256 --pretrain_epoch 300 --pretrain_lr 0.001 --noise 0.2\
    --pretrain_optim Adam --net VAE --channel 1 --wide 32 --num_workers 0\
    --cluster_batchsize 1000 --cluster_epoch 500 --cluster_lr 0.01 --cluster_optim Adam\
    --manifold_lr 0.01 --change 30 --m 10 --divide 0.03 --total 60000',

    'python two_stage.py --name Cifar_10 --input_size 3072 --n_cluster 10 --hidden_dim 10 \
    --pretrain_batchsize 256 --pretrain_epoch 350 --pretrain_lr 0.0001 --noise 0.2\
    --pretrain_optim Adam --net VAE --channel 1 --wide 32 --num_workers 0\
    --cluster_batchsize 1000 --cluster_epoch 500 --cluster_lr 0.001 --cluster_optim Adam\
    --manifold_lr 0.0001 --change 30 --m 10 --divide 0 --total 60000 --Acc 26.73',

    'python two_stage.py --name Cifar_10 --input_size 3072 --n_cluster 10 --hidden_dim 10 \
    --pretrain_batchsize 256 --pretrain_epoch 300 --pretrain_lr 0.0001 --noise 0.2\
    --pretrain_optim Adam --net VAE --channel 1 --wide 32 --num_workers 0\
    --cluster_batchsize 1000 --cluster_epoch 500 --cluster_lr 0.001 --cluster_optim Adam\
    --manifold_lr 0.0001 --change 30 --m 10 --divide 0 --total 60000 --Acc 27.59',

    'python two_stage.py --name tiny --input_size 12288 --n_cluster 200 --hidden_dim 200\
    --pretrain_batchsize 256 --pretrain_epoch 200 --pretrain_lr 0.0001 --noise 0.2\
    --pretrain_optim Adam --net VAE --channel 1 --wide 64 --num_workers 0\
    --cluster_batchsize 1000 --cluster_epoch 200 --cluster_lr 0.001 --cluster_optim Adam\
    --manifold_lr 0.0001 --change 30 --m 10 --divide 0 --total 60000 --Acc 26.73',

    'python two_stage.py --name STL_10 --input_size 27648 --n_cluster 10 --hidden_dim 10\
    --pretrain_batchsize 256 --pretrain_epoch 400 --pretrain_lr 0.0001 --noise 0.2\
    --pretrain_optim Adam --net VAE --channel 1 --wide 96 --num_workers 0\
    --cluster_batchsize 1000 --cluster_epoch 500 --cluster_lr 0.001 --cluster_optim Adam\
    --manifold_lr 0.0001 --change 30 --m 10 --divide 0 --total 60000 --Acc 26.73',

    'python two_stage.py --name Mnist --input_size 784 --n_cluster 10 --hidden_dim 10\
    --pretrain_batchsize 256 --pretrain_epoch 200 --pretrain_lr 0.001 --noise 0.1\
    --pretrain_optim Adam --net VAE --channel 1 --wide 28 --num_workers 0\
    --cluster_batchsize 1000 --cluster_epoch 8000 --cluster_lr 0.001 --cluster_optim Adam\
     --manifold_lr 0.001 --change 2 --m 10 --divide 0.1 --total 70000 --Acc 77.48 --times 700',

    'python two_stage.py --name Mnist --input_size 784 --n_cluster 10 --hidden_dim 10\
    --pretrain_batchsize 256 --pretrain_epoch 200 --pretrain_lr 0.001 --noise 0.1\
    --pretrain_optim Adam --net VAE --channel 1 --wide 28 --num_workers 0 \
    --cluster_batchsize 1000 --cluster_epoch 8000 --cluster_lr 0.001 --cluster_optim Adam\
     --manifold_lr 0.001 --change 2 --m 10 --divide 0.1 --total 70000 --Acc 77.48 --times 701',

    'python two_stage.py --name Mnist --input_size 784 --n_cluster 10 --hidden_dim 10\
    --pretrain_batchsize 256 --pretrain_epoch 200 --pretrain_lr 0.001 --noise 0.1\
    --pretrain_optim Adam --net VAE --channel 1 --wide 28 --num_workers 0 \
    --cluster_batchsize 1000 --cluster_epoch 8000 --cluster_lr 0.001 --cluster_optim Adam\
     --manifold_lr 0.001 --change 2 --m 10 --divide 0.1 --total 70000 --Acc 77.48 --times 702',

    'python two_stage.py --name USPS2 --input_size 256 --n_cluster 10 --hidden_dim 10 \
    --pretrain_batchsize 256 --pretrain_epoch 300 --pretrain_lr 0.001 --noise 0.1\
    --pretrain_optim Adam --net VAE --channel 1 --wide 16 --num_workers 0 \
    --cluster_batchsize 1000 --cluster_epoch 400 --cluster_lr 0.001 --cluster_optim Adam\
    --manifold_lr 0.001 --change 30 --m 10 --divide 0.1 --total 9298 --Acc 73.33 --times 10015',

    'python two_stage.py --name Mnist --input_size 784 --n_cluster 10 --hidden_dim 10\
    --pretrain_batchsize 256 --pretrain_epoch 500 --pretrain_lr 0.001 --noise 0.1\
    --pretrain_optim Adam --net VAE --channel 1 --wide 28 --num_workers 0 \
    --cluster_batchsize 1000 --cluster_epoch 400 --cluster_lr 0.001 --cluster_optim Adam\
     --manifold_lr 0.001 --change 2 --m 10 --divide 0.1 --total 70000 --Acc 81.5 --times 11015',

    'python two_stage.py --name Fashion2 --input_size 784 --n_cluster 10 --hidden_dim 10\
    --pretrain_batchsize 256 --pretrain_epoch 500 --pretrain_lr 0.001 --noise 0.3\
    --pretrain_optim Adam --net VAE --channel 1 --wide 28 --num_workers 0 \
    --cluster_batchsize 1000 --cluster_epoch 400 --cluster_lr 0.001 --cluster_optim Adam\
    --manifold_lr 0.001 --change 2 --m 10 --divide 0 --total 70000 --Acc 58.19 --times 12015',
]

path = 'python '

cmd_run = [
    'pretrain.py --name Reuters_10K --input_size 2000 --n_cluster 4 --hidden_dim 10 --shuffle --pretrain \
    --pretrain_batchsize 256 --pretrain_epoch 500 --pretrain_lr 0.0001 --noise 0.2 --trans\
    --pretrain_optim Adam --net VAE --channel -1 --wide -1 --num_workers 0',
]
cmd_cluster_our = [
    'two_stage.py --name Reuters_10K --input_size 2000 --n_cluster 4 --hidden_dim 10\
    --pretrain_batchsize 256 --pretrain_epoch 200 --pretrain_lr 0.0001 --noise 0.2\
    --pretrain_optim Adam --net VAE --channel -1 --wide -1 --num_workers 0\
    --cluster_batchsize 1000 --cluster_epoch 5000 --cluster_lr 0.001 --cluster_optim Adam\
    --manifold_lr 0.0001 --change 2 --m 10 --divide 0.1 --total 70000 --Acc 67.34 --times 101',
]

#cmd = cmd_run
cmd = cmd_cluster_our

for i in range(len(cmd)):
    print(i)
    os.system(path+cmd[i])
