#! /bin/bash

python main.py /home/dataset/ILSVRC/Data/CLS-LOC --arch vgg13 --pretrained --evaluate --batch-size 128 --keep-logs --random-resize 224 256 288
clear
python main.py /home/dataset/ILSVRC/Data/CLS-LOC --arch vgg16 --pretrained --evaluate  --batch-size 128 --keep-logs --random-resize 224 256 288
clear
python main.py /home/dataset/ILSVRC/Data/CLS-LOC --arch vgg19 --pretrained --evaluate  --batch-size 128 --keep-logs --random-resize 224 256 288
