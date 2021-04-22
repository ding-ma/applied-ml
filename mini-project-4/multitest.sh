#! /bin/bash

python main.py --arch vgg11 --pretrained --evaluate --keep-logs /home/dataset/ILSVRC/Data/CLS-LOC
clear
python main.py --arch vgg13 --pretrained --evaluate --keep-logs /home/dataset/ILSVRC/Data/CLS-LOC
clear
python main.py --arch vgg16 --pretrained --evaluate --keep-logs /home/dataset/ILSVRC/Data/CLS-LOC
clear
python main.py --arch vgg19 --pretrained --evaluate --keep-logs /home/dataset/ILSVRC/Data/CLS-LOC
