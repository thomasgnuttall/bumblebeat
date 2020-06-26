import click

from bumblebeat.data import data_main
from bumblebeat.utils.data import load_yaml

import tensorflow_datasets as tfds
from magenta import music as mm
import click

#from bumblebeat.data import data_main
from bumblebeat.utils.data import load_yaml, get_bucket_number
from bumblebeat.data import get_corpus
from bumblebeat.output.generate import *

conf_path = 'conf/train_conf.yaml'
conf = load_yaml(conf_path)

pitch_classes = load_yaml('conf/drum_pitches.yaml')
time_steps_vocab = load_yaml('conf/time_steps_vocab.yaml')


model_conf = conf['model']
data_conf = conf['data']

corpus = get_corpus(
    data_conf['dataset'],
    data_conf['data_dir'],
    pitch_classes['DEFAULT_DRUM_TYPE_PITCHES'],
    time_steps_vocab,
    conf['processing']
)
dataset = tfds.as_numpy(
    tfds.load(
        name='groove/full-midionly',
        split='train',
        try_gcs=True
    ))

dev_sequences = [mm.midi_to_note_sequence(features["midi"]) for features in dataset]
dev_sequences = [corpus._quantize(d, 4) for d in dev_sequences]

simplified_pitches = [[36], [38], [42], [46], [45], [48], [50], [49], [51]]

original = dev_sequences[0]
tokens = corpus._tokenize(original, 4, True)

import click

#from bumblebeat.data import data_main
from bumblebeat.utils.data import load_yaml
from bumblebeat.data import get_corpus

conf_path = 'conf/train_conf.yaml'
conf = load_yaml(conf_path)


pitch_classes = load_yaml('conf/drum_pitches.yaml')
time_vocab = load_yaml('conf/time_steps_vocab.yaml')


model_conf = conf['model']
data_conf = conf['data']

corpus = get_corpus(
    data_conf['dataset'],
    data_conf['data_dir'],
    pitch_classes['DEFAULT_DRUM_TYPE_PITCHES'],
    time_steps_vocab,
    conf['processing']
)
pitch_vocab = corpus.reverse_vocab

path = 'gpu_run-groove/full-midionly/20200624-191931/train_step_120015/model.pt'
USE_CUDA = False
tgt_len = 1
ext_len = 0
mem_len = 2000
clamp_len = 1000
gen_len = 1000
same_len = True

hat_prime=[95,2,2,2,2,1,1,1,1,1,1,1,1,42,2,2,2,2,1,1,1,1,1,1,1,1,42,2,2,2,2,1,1,1,1,1,1,1,1,42,2,2,2,2,1,1,1,1,1,1,1,1,42,2,2,2,2,1,1,1,1,1,1,1,1,42,2,2,2,2,1,1,1,1,1,1,1,1,42]
simplified_pitches = [[36], [38], [42], [46], [45], [48], [50], [49], [51]]
device = torch.device("cuda" if USE_CUDA else "cpu")

model = load_model(path, device)

samplefrom = tokens

temp = 0.95
topk = 32
memlen = 512

temp = 0.96
topk = 64

def continue_sequence(model, seq, prime_len, gen_len, temp, topk, mem_len, device):


