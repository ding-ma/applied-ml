# ML Reproducibility Challenge
We aim to reproduce: __Very Deep Convolutional Networks for Large-Scale Image Recognition__ by Karen Simonyan, and Andrew Zisserman. The paper can be found [here](https://arxiv.org/abs/1409.1556)

* Report [link](https://www.overleaf.com/1897293349fjsssxwrwydb)

## Dependencies
* Google Cloud Deep Learning VM with Pytorch 1.8. You should see in your terminal the `(base)` conda environment
```bash
(base) ding@deeplearning-4-vm:~/applied-ml/mini-project-4/$
``` 
* If there are additional packages, install with `pip install -r requirements.txt`

## PreTrained Models
* See [PyTorch](https://pytorch.org/vision/stable/models.html). The models are trained on the ImageNet Dataset. They are trained on [224x224 images](https://discuss.pytorch.org/t/imagenet-pretrained-models-image-dimensions/94649)


## Running the model
We have modified the [Pytorch implementation](https://github.com/pytorch/examples/tree/master/imagenet) to be able to tweak hyperparamters and log the output to a file.

Modified:
* `--batch-size`: defaults to 128 instead of 256 due to GPU memory limitations. Only images sized at 256x256 can use a batch size of 256, otherwise the program will crash.
* `--arch`: Takes in _custom_ as an argument. It allows the user to modify the `customVGG.py`class. 

Added params:
* `--img`: size of the image (default to 256). 
* `--no-normalize`: Remove image color normalization.
* `--keep-logs`: save logs to a file when training or testing. The output are saved to `./logs`
* `--jitter-val`: randomly resize validation images. The smaller images will be padded with 0 to match the largest one. E.g. `--jitter-val 224 256 288`
* `--jitter-train`: randomly resize training images. The smaller images will be padded with 0 to match the largest one.

Example: `python main.py /home/dataset/ILSVRC/Data/CLS-LOC --arch vgg11 --pretrained --evaluate`

## Dataset Location
* Download from [Imaget 2012](http://image-net.org/challenges/LSVRC/2012/2012-downloads): 
1. Test set (15GB): 100,000 images. Labels are NOT available
1. Validation set (5GB): 50,000 images. Labels are located at: `/home/dataset/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt`
1. Train set (124GB): 1.2M images. The file name contains the label.
```
wget -d --header="X-Auth-Token: your_access_token" url
```

**Warning** This will take a few hours. 

* The dataset is located in `/home/dataset/ILSVRC/`
* If you encounter any issue with permssion, run `sudo chmod -R 777 /home/dataset`

## GPU for project
Make sure when you run `nvidia-smi`, you see the GPU you attached.
```console
(base) ding@instance-1:~/applied-ml/mini-project-4/$ nvidia-smi
Sat Apr 24 01:14:48 2021       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.80.02    Driver Version: 450.80.02    CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla P100-PCIE...  Off  | 00000000:00:04.0 Off |                    0 |
| N/A   73C    P0   182W / 250W |  14459MiB / 16280MiB |    100%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A     10745      C   python                          14453MiB |
+-----------------------------------------------------------------------------+
```

To continously watch: `watch -n0.1 nvidia-smi`


## Todos
* > You will first reproduce the results reported in the paper by running the code provided by the authors or by implementing on your own, if no code is available
* > You will try to modify the model and perform ablation studies to understand the model’s robustness and evaluate the importance of the various model components. (In this context, the term “ablation” is used to describe the process of removing different model components to see how it impacts performance.)
* > You should do a thorough analysis of the model through an extensive set of experiments.
* > Note that some experiments will be difficult to replicate due to computational resources. It is fine to reproduce only a subset of the original paper’s results or to work on a smaller variant of the data—if necessary.
* > At a minimum, you should use the authors code to reproduce a non-trivial subset of their results and explore how the model performs after you make minor modifications (e.g., changes to hyperparameters).
* > An outstanding project would perform a detailed ablation study and/or implement significant/meaningful extensions of the model.
