#! /bin/bash


python main.py /home/dataset/ILSVRC/Data/CLS-LOC --batch-size 256 --arch vgg11 --pretrained --evaluate --keep-logs --no-normalize
clear
python main.py /home/dataset/ILSVRC/Data/CLS-LOC --batch-size 256 --arch vgg16 --pretrained --evaluate --keep-logs --no-normalize
clear
python main.py /home/dataset/ILSVRC/Data/CLS-LOC --batch-size 256 --arch vgg19 --pretrained --evaluate --keep-logs --no-normalize

clear
python main.py /home/dataset/ILSVRC/Data/CLS-LOC --batch-size 256 --arch vgg11_bn --pretrained --evaluate --keep-logs --no-normalize
clear
python main.py /home/dataset/ILSVRC/Data/CLS-LOC --batch-size 256 --arch vgg16_bn --pretrained --evaluate --keep-logs --no-normalize
clear
python main.py /home/dataset/ILSVRC/Data/CLS-LOC --batch-size 256 --arch vgg19_bn --pretrained --evaluate  --keep-logs --no-normalize