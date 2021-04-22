# ML Reproducibility Challenge
We aim to reproduce: __Very Deep Convolutional Networks for Large-Scale Image Recognition__ by Karen Simonyan, and Andrew Zisserman. The paper can be found [here](https://arxiv.org/abs/1409.1556)

* Report [link](https://www.overleaf.com/project/607ce84766796508c3be39e9)

## Dependencies
* Google Cloud Deep Learning VM with Pytorch 1.8. You should see `(base) ding@deeplearning-4-vm:~/applied-ml/mini-project-4/$` in your terminal.
* If there are additional packages, install with `pip install -r requirements.txt`

## PreTrained Models
* See [PyTorch](https://pytorch.org/vision/stable/models.html). The models are trained on the ImageNet Dataset. They are trained by [224x224 images](https://discuss.pytorch.org/t/imagenet-pretrained-models-image-dimensions/94649)


## Running the model
We have modified the [Pytorch implementation](https://github.com/pytorch/examples/tree/master/imagenet) to be able to tweak hyperparamters and log the output to a file.

Added params:
* `--img`: size of the image (default to 256). A size of 384 present in the paper needs a reduction of the batch size to 128
* `--normalize`: normalization of the RGB values (default to true)
* `--keep-logs`: save logs to a file when training or testing

Example: `python main.py --arch vgg11 --pretrained --evaluate /home/dataset/ILSVRC/Data/CLS-LOC`

## Dataset Location
* Download from [Imaget 2012](http://image-net.org/challenges/LSVRC/2012/2012-downloads): 
1. Test set (15GB): 100,000 images. Labels are NOT available
1. Validation set (5GB): 50,000 images. Labels are located at: `/home/dataset/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt`
1. Train set (124GB): 1.2M images. The file name contains the label.
```
wget -d --header="X-Auth-Token: your_access_token" url
```
* All dataset are located in `/home/dataset`
* If you encounter any issue with permssion, run `sudo chmod -R 777 /home/dataset`

## GPU for project
Make sure when you run `nvidia-smi`, you see the GPU you attached.
```console
ding@deeplearning-3-vm:~/applied-ml/mini-project-4/$ nvidia-smi
Sun Apr 18 20:11:04 2021       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.80.02    Driver Version: 450.80.02    CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla K80           Off  | 00000000:00:04.0 Off |                    0 |
| N/A   43C    P0    66W / 149W |      0MiB / 11441MiB |    100%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
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