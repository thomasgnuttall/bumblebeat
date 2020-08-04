# Script for initialising Deep Learning Base AMI (Amazon Linux 2) Version 50.0
# Use the fswatch trick from scripts/instance.sh to ensure bumblebeat/ exists on instance

############################################
# To save session in case of lost connection
############################################
sudo yum install tmux
tmux
source activate pytorch_latest_p36


#############
# Virtual Env
#############
# NOT NEEDED IN V 50.0 (virtual envs exist on instance)
#sudo pip install virtualenvwrapper
#export WORKON_HOME=~/envs
#
#pip install python3
#export VIRTUALENVWRAPPER_PYTHON='/usr/bin/python3.7'
#mkdir -p $WORKON_HOME
#source /usr/local/bin/virtualenvwrapper.sh
#mkvirtualenv bumblebeat --python='/usr/bin/python3.7'


#################################
# Magenta/Bumblebeat dependencies
#################################
sudo yum install gcc-c++ git alsa-lib-devel
sudo cp /etc/ssl/certs/ca-bundle.crt /etc/ssl/certs/ca-certificates.crt


#########
# Package
#########
cd bumblebeat
pip install -e .


######## CUDA and GPUs #########
## GPU info
#lspci | grep -i nvidia # = 00:1e.0 3D controller: NVIDIA Corporation GK210GL [Tesla K80] (rev a1) on EC2 p2.xlarge
# 
#
## To get kernel version of system
#uname -r # = 4.14.177-139.254.amzn2.x86_64
#sudo yum install kernel-devel-$(uname -r) kernel-headers-$(uname -r)


#   $ nvidia-smi
#
#   +-----------------------------------------------------------------------------+
#   | NVIDIA-SMI 440.33.01    Driver Version: 440.33.01    CUDA Version: 10.2     |
#   |-------------------------------+----------------------+----------------------+
#   | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
#   | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
#   |===============================+======================+======================|
#   |   0  Tesla K80           On   | 00000000:00:1E.0 Off |                    0 |
#   | N/A   40C    P8    26W / 149W |      0MiB / 11441MiB |      0%      Default |
#   +-------------------------------+----------------------+----------------------+
#
#   +-----------------------------------------------------------------------------+
#   | Processes:                                                       GPU Memory |
#   |  GPU       PID   Type   Process name                             Usage      |
#   |=============================================================================|
#   |  No running processes found                                                 |
#   +-----------------------------------------------------------------------------+