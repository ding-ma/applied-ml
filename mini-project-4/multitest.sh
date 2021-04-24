#! /bin/bash


python main.py /home/dataset/ILSVRC/Data/CLS-LOC --batch-size 256 --arch vgg11_bn --pretrained --evaluate --keep-logs --img 256
clear
python main.py /home/dataset/ILSVRC/Data/CLS-LOC --batch-size 256 --arch vgg13_bn --pretrained --evaluate --keep-logs --img 256
clear
python main.py /home/dataset/ILSVRC/Data/CLS-LOC --batch-size 256 --arch vgg16_bn --pretrained --evaluate --keep-logs --img 256
clear
python main.py /home/dataset/ILSVRC/Data/CLS-LOC --batch-size 256 --arch vgg19_bn --pretrained --evaluate --keep-logs --img 256
clear

python main.py /home/dataset/ILSVRC/Data/CLS-LOC --batch-size 128 --arch vgg13_bn --pretrained --evaluate --keep-logs --img 384
clear
python main.py /home/dataset/ILSVRC/Data/CLS-LOC --batch-size 128 --arch vgg16_bn --pretrained --evaluate --keep-logs --img 384
clear
python main.py /home/dataset/ILSVRC/Data/CLS-LOC --batch-size 128 --arch vgg19_bn --pretrained --evaluate --keep-logs --img 384
clear

########## multi scale eval
python main.py /home/dataset/ILSVRC/Data/CLS-LOC --batch-size 128 --arch vgg13_bn --pretrained --evaluate --keep-logs --jitter-val 224 256 288
clear
python main.py /home/dataset/ILSVRC/Data/CLS-LOC --batch-size 128 --arch vgg16_bn --pretrained --evaluate --keep-logs --jitter-val 224 256 288
clear
python main.py /home/dataset/ILSVRC/Data/CLS-LOC --batch-size 128 --arch vgg19_bn --pretrained --evaluate --keep-logs --jitter-val 224 256 288
clear

python main.py /home/dataset/ILSVRC/Data/CLS-LOC --batch-size 64 --arch vgg16_bn --pretrained --evaluate --keep-logs --jitter-val 256 384 512
clear
python main.py /home/dataset/ILSVRC/Data/CLS-LOC --batch-size 64 --arch vgg19_bn --pretrained --evaluate --keep-logs --jitter-val 256 384 512
clear
