import os
import sys
sys.path.append(r'\home\ubuntu\workspace')


path = 'python '

cmd_cluster_Mnist_CNN = [
    'two_stage.py                                                                           \
    --name             Mnist         --n_cluster        10        --num_workers       5     \
    --optim            Adam          --net              CNN_VAE3  --channel           3     \
    --wide             224           --m                10        --divide            0.1   \
    --reload           False         --hidden_dim       10       --times             0       \
    --Intra_batchsize  200           --Inter_batchsize  300       --global_batchsize  300   \
    --Intra_epoch      0             --Inter_epoch      0         --global_epoch      0    \
    --Intra_lr         0.0003        --Inter_lr         0.0001    --global_lr         0.0001 ',
]

cmd_cluster_Mnist_fanhu_CNN = [
    'two_stage_fanhua.py      --nick_name   Ours                                             \
    --name             test_mnist         --n_cluster        10        --num_workers       5     \
    --optim            Adam          --net              CNN_VAE3  --channel           3     \
    --wide             224           --m                10        --divide            0.1   \
    --reload           True         --hidden_dim       10        --times               0   \
    --Intra_batchsize  200           --Inter_batchsize  300       --global_batchsize  300   \
    --Intra_epoch      100             --Inter_epoch      0         --global_epoch      0     \
    --Intra_lr         0.0003        --Inter_lr         0.0001    --global_lr         0.0001 \
    --Intra_Max        101           --Inter_Max        101       --Global_Max        101   ',
]

cmd_cluster_Mnist_test_CNN = [
    'two_stage.py      --nick_name   Ours                                             \
    --name             Mnist         --n_cluster        10        --num_workers       5     \
    --optim            Adam          --net              CNN_VAE3  --channel           3     \
    --wide             224           --m                10        --divide            0.1   \
    --reload           True         --hidden_dim       10        --times               0   \
    --Intra_batchsize  200           --Inter_batchsize  300       --global_batchsize  300   \
    --Intra_epoch      100             --Inter_epoch      0         --global_epoch      0     \
    --Intra_lr         0.0003        --Inter_lr         0.0001    --global_lr         0.0001 \
    --Intra_Max        101           --Inter_Max        101       --Global_Max        101   ',
]

cmd_cluster_Mnist_L1_L3_CNN = [
    'two_stage_L1+L3.py      --nick_name   Ours                                             \
    --name             Mnist         --n_cluster        10        --num_workers       5     \
    --optim            Adam          --net              CNN_VAE3  --channel           3     \
    --wide             224           --m                10        --divide            0.1   \
    --reload           True         --hidden_dim       10        --times               0   \
    --Intra_batchsize  200           --Inter_batchsize  300       --global_batchsize  300   \
    --Intra_epoch      100             --Inter_epoch      0         --global_epoch      0     \
    --Intra_lr         0.0003        --Inter_lr         0.0001    --global_lr         0.0001 \
    --Intra_Max        101           --Inter_Max        101       --Global_Max        101   ',
]

cmd_cluster_Mnist_DEC = [
    'two_stage_DEC_NEW.py      --nick_name   Ours                                             \
    --name             Mnist         --n_cluster        10        --num_workers       5     \
    --optim            Adam          --net              CNN_VAE3  --channel           3     \
    --wide             224           --m                10        --divide            0.1   \
    --reload           True         --hidden_dim       10        --times               0   \
    --Intra_batchsize  200           --Inter_batchsize  300       --global_batchsize  300   \
    --Intra_epoch      100             --Inter_epoch      100         --global_epoch      0     \
    --Intra_lr         0.0003        --Inter_lr         0.0001    --global_lr         0.0001 \
    --Intra_Max        101           --Inter_Max        101       --Global_Max        101   ',
]

#cmd = cmd_pretrain
#cmd = cmd_cluster_Mnist_test_CNN
cmd = cmd_cluster_Mnist_DEC
#cmd = cmd_cluster_Mnist_canshu

for i in range(len(cmd)):
    print(i)
    os.system(path+cmd[i])
