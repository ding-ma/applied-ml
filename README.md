# applied-ml
COMP 551 - Applied Machine Learning (Winter 2021)

## To SSH onto the VM
1. Create your ssh public and private key.
1. Upload your public ssh key to GCP
1. `ssh -i path_to_private_key username@34.122.237.6`. The IP address of the VM is static.


## Create virtual environment
1. `pip3 install virtualenv`, if you do not have it yet
1. `virtualenv venv`
1. `source venv/bin/activate`, you should see `venv` in front of your terminal prompt
1. `pip install -r requirements.txt`


## Reading and writing files
All the dataset are located at `/home/dataset`. There is a dataset folder for each project. 
* If there is a permission denied issue, make sure to run `sudo chmod -R 777 /home/dataset`. 
* If you create new dataset files, run `sudo chmod -R 777 /home/dataset`.