import click

from bumblebeat.data import data_main
from bumblebeat.utils.data import load_yaml


conf_path = 'conf/train_conf.yaml'
conf = load_yaml(conf_path)

pitch_classes = load_yaml('conf/drum_pitches.yaml')
time_steps_vocab = load_yaml('conf/time_steps_vocab.yaml')


model_conf = conf['model']
data_conf = conf['data']
model_conf['cuda']= True
model_conf['restart'] = False

#LOOK INTO TIED/TIE_PROJ


#---------------------------------------------------------------------------
#AttributeError                            Traceback (most recent call last)
#<ipython-input-5-3ce0f28dd575> in <module>
#    152 try:
#    153     for epoch in itertools.count(start=1):
#--> 154         train()
#    155         if train_step == model_conf['max_step']:
#    156             logging('-' * 100)

#<ipython-input-5-3ce0f28dd575> in train()
#     41         mems = tuple()
#     42     train_iter = tr_iter.get_varlen_iter() if model_conf['varlen'] else tr_iter
#---> 43     for batch, (data, target, seq_len) in enumerate(train_iter):
#     44         model.zero_grad()
#     45         if model_conf['batch_chunk'] > 1:

#~/bumblebeat/bumblebeat/data.py in get_varlen_iter(self, start, std, min_len, max_deviation)
#    188             bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
#    189             bptt = min(max_len, max(min_len, int(np.random.normal(bptt, std))))
#--> 190             data, target, seq_len = self.get_batch(i, bptt)
#    191             i += seq_len
#    192             yield data, target, seq_len

#~/bumblebeat/bumblebeat/data.py in get_batch(self, i, bptt)
#    168     def get_batch(self, i, bptt=None):
#    169         if bptt is None: bptt = self.bptt
#--> 170         seq_len = min(bptt, self.data.size(0) - 1 - i)
#    171
#    172         end_idx = i + seq_len

#AttributeError: 'list_iterator' object has no attribute 'size'