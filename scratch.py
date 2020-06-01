import click

from bumblebeat.data import data_main
from bumblebeat.utils.data import load_yaml

conf = load_yaml(conf_path)
conf_path = 'conf/train_conf.yaml'
pitch_classes = load_yaml('conf/drum_pitches.yaml')
time_steps_vocab = load_yaml('conf/time_steps_vocab.yaml')


model_conf = conf['model']
data_conf = conf['data']
model_conf['cuda']= True
model_conf['restart'] = False

#LOOK INTO TIED/TIE_PROJ