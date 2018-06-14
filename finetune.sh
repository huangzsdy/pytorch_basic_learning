source /home/huangzesang/share/env_python3.6.sh
# CIFAR10=/home/huangzesang/data/CIFAR10/png
# CLASSES=('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

TRAIN_LIST=/home/safe_data_dir_2/huangzesang/dataset/small_imagenet/train.txt
TEST_LIST=/home/safe_data_dir_2/huangzesang/dataset/small_imagenet/val.txt
BATCH_SIZE=10
RESUME=checkpoints/finetune_train_600.pth
PROCESS_NAME=finetune_train
FT_NET=Resnet18
SNAPSHOT=100

python -u finetune_imagenet.py \
        --train_list ${TRAIN_LIST}\
        --test_list ${TEST_LIST}\
        --batch_size ${BATCH_SIZE}\
        --proc_name ${PROCESS_NAME}\
        --no_cuda 0\
        --ft_net ${FT_NET}\
        --snapshot ${SNAPSHOT}\
        # --resume ${RESUME}\

                # 2>&1|tee logs/train.log &

