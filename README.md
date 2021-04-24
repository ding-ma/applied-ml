# Applied-ML
COMP 551 - Applied Machine Learning (Winter 2021)

## Projects
1. KNN and Decision Tree on cancer data
1. Naive Bayes and Linear Regression on text data
1. Multilayer Perceptron on MNIST
1. YOLO: Unified, Real-Time Object Detection Reproducibility Challenge

## To SSH onto the VM
1. Create your ssh public and private key.
1. Upload your public ssh key to GCP
1. `ssh -i path_to_private_key username@34.74.0.92`. The IP address of the VM is static.

**NOTE**: IP of VM changed since mini-project-3

## Create virtual environment
Each mini-project has its own virtual environment
1. `cd` into the mini-project you want to work with
1. Create your virtualenv: `python -m venv venv`
1. `source venv/bin/activate`, you should see `venv` in front of your terminal prompt
1. Install project dependencies: `pip install -r requirements.txt`
1. If you installed new dependencies: `pip freeze > requirements.txt`

Note: Not needed for mini-project 4. We will use the conda base package from GCP Deep Learning VM

## Running a task in background
1. `nohup python script_name.py &`
1. Close your terminal and go Zzzz