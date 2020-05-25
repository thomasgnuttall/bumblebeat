# Script for initialising Amazon Linux AMI 2018.03.0.20200514.0 x86_64 HVM 

# To save session in case of lost connection
sudo yum install tmux
tmux

# Virtual Env
pip-3.6 install virtualenvwrapper
export WORKON_HOME=~/envs
export VIRTUALENVWRAPPER_PYTHON='/usr/bin/python3'
mkdir -p $WORKON_HOME
source /usr/local/bin/virtualenvwrapper.sh
mkvirtualenv bumblebeat --python='/usr/bin/python3'

# Magenta/Bumblebeat dependencies
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
export PATH=/home/linuxbrew/.linuxbrew/bin:$PATH
brew install libsndfile
# TODO: FIX THIS
# At the moment Magenta fails because of libsndfile
# commenting out the unused import Librosa in /home/ec2-user/envs/bumblebeat/lib/python3.6/site-packages/magenta/music/audio_io.py
# solves the issue
sudo yum install gcc-c++
sudo yum install -y python36-devel.x86_64
sudo yum install alsa-lib-devel
sudo yum install git
sudo yum install python36 python36-pip

git clone https://github.com/slimranking/bumblebeat.git
cd bumblebeat
pip install -e .
