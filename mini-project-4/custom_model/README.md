# Custom Models

In here, we modified the VGG implementation to see how the model learns. All model files are hosted on GitHub as a release binary as the files are too big to be committed directly.

The training logs are very big. To see the test accuracy between the epochs, run `cat nohup.out | grep "* Acc"`

## Experiment 1
We decided to remove the last conv3-512 block from VGG 11 which gives us a "_VGG 9_". The model is the following:
```python
    VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): ReLU(inplace=True)
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(inplace=True)
    (8): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU(inplace=True)
    (10): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (11): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (12): ReLU(inplace=True)
    (13): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (14): ReLU(inplace=True)
    (15): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
```
~~The training of this model crashed during its 13th epoch.~~ We reattempted training of this model with a `learning rate: 0.001` and we got much better results. 

```
2021-04-27 04:16:14 INFO      * Acc@1 47.298 Acc@5 73.276
2021-04-27 05:52:55 INFO      * Acc@1 52.112 Acc@5 77.188
2021-04-27 07:29:39 INFO      * Acc@1 54.700 Acc@5 78.984
2021-04-27 09:06:23 INFO      * Acc@1 55.928 Acc@5 79.934
2021-04-27 10:43:08 INFO      * Acc@1 56.872 Acc@5 80.758
2021-04-27 12:19:51 INFO      * Acc@1 58.350 Acc@5 81.626
2021-04-27 13:56:39 INFO      * Acc@1 58.310 Acc@5 81.558
2021-04-27 15:33:24 INFO      * Acc@1 59.566 Acc@5 82.352
2021-04-27 17:10:12 INFO      * Acc@1 59.630 Acc@5 82.368
2021-04-27 18:46:59 INFO      * Acc@1 59.890 Acc@5 82.770
2021-04-27 20:24:07 INFO      * Acc@1 60.210 Acc@5 83.006
2021-04-27 22:00:55 INFO      * Acc@1 60.312 Acc@5 83.004
2021-04-27 23:37:43 INFO      * Acc@1 61.000 Acc@5 83.428
2021-04-28 01:14:33 INFO      * Acc@1 61.176 Acc@5 83.716
```




## Experiment 2
We changed the kernel size to 5 while using VGG11. The model is the following:
```python
    VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(64, 128, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1))
    (4): ReLU(inplace=True)
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(128, 256, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1))
    (7): ReLU(inplace=True)
    (8): Conv2d(256, 256, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1))
    (9): ReLU(inplace=True)
    (10): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (11): Conv2d(256, 512, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1))
    (12): ReLU(inplace=True)
    (13): Conv2d(512, 512, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1))
    (14): ReLU(inplace=True)
    (15): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (16): Conv2d(512, 512, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1))
    (17): ReLU(inplace=True)
    (18): Conv2d(512, 512, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1))
    (19): ReLU(inplace=True)
    (20): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
```
Outcome: Not very successful and very time consuming.


## Experiment 3
Trained on VGG 11 with image size of 128x128. Started with the pretrained model and finue-tuned for 10 epochs.
Results
```
2021-04-26 16:36:54 INFO      * Acc@1 59.418 Acc@5 82.264
2021-04-26 17:43:53 INFO      * Acc@1 60.112 Acc@5 82.742
2021-04-26 18:49:15 INFO      * Acc@1 60.582 Acc@5 83.220
2021-04-26 19:52:42 INFO      * Acc@1 60.924 Acc@5 83.476
2021-04-26 20:56:45 INFO      * Acc@1 61.160 Acc@5 83.564
2021-04-26 22:00:25 INFO      * Acc@1 61.502 Acc@5 83.822
2021-04-26 23:03:43 INFO      * Acc@1 61.734 Acc@5 83.898
2021-04-27 00:07:58 INFO      * Acc@1 61.728 Acc@5 84.052
2021-04-27 01:11:57 INFO      * Acc@1 62.024 Acc@5 84.182
2021-04-27 02:16:24 INFO      * Acc@1 62.090 Acc@5 84.206
```


