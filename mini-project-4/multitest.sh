#! /bin/bash


python main.py /home/dataset/ILSVRC/Data/CLS-LOC --batch-size 64 --arch vgg19 --pretrained --evaluate --keep-logs --jitter-val 256 384 512
