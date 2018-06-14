source /home/huangzesang/share/env_python3.6.sh
CIFAR10=/home/huangzesang/data/CIFAR10/png
BATCH_SIZE=10
RESUME=work/densenet.base/latest.pth
PROCESS_NAME=densenet_train
CLASSES=('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

python -u train.py \
        --dataPath ${CIFAR10}\
        --batchSz ${BATCH_SIZE}\
        --procName ${PROCESS_NAME}\
        --no_cuda 0\
        # --resume ${RESUME}\

                # 2>&1|tee logs/train.log &

