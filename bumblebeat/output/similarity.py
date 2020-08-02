import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.manifold import TSNE

from magenta import music as mm

from bumblebeat.utils.generation import TxlSimpleSampler
from bumblebeat.output.generate import prime_sampler

#from bumblebeat.data import data_main
from bumblebeat.utils.data import load_yaml
from bumblebeat.output.colours import cnames

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

valid_data = corpus.valid_data

genre_lookup = {
    0: 'afrobeat',
    1: 'afrocuban',
    2: 'blues',
    3: 'country',
    4: 'dance',
    5: 'funk',
    6: 'gospel',
    7: 'highlife',
    8: 'hiphop',
    9: 'jazz',
    10: 'latin',
    11: 'middleeastern',
    12: 'neworleans',
    13: 'pop',
    14: 'punk',
    15: 'reggae',
    16: 'rock',
    17: 'soul'}

lookup_genre = {v:k for k,v in genre_lookup.items()}

def get_feature_frame(dataset, genre_lookup, corpus, model, mem_len=512, prime_len=512, quantize=True, steps_per_quarter=4, filter_4_4=False):
    """
    From <dataset> extract feature frame of metadata and embedding
    
    Param
    =====
    dataset: list
        List of datapoints from tf.datasets
    genre_lookup: dict
        dict of {style_id:genre}
    corpus: corpus
        Corpus with tokeniser
    model: model
        Trained transformer model (on corpus)
    mem_len: int
        Memory length of the model
    prime_len: int
        How many of the most recent tokens to prime model with before
        extracting embedding (if None use whole sequence)
    quantize: bool
        Quantize sequence before tokenisation?
    steps_per_quarter: int
        Step resolution of <quantize>
    filter_4_4: bool
        Filter to include only 4/4 time sigs?
    """
    embeddings = get_embeddings(
        dataset, corpus, model, mem_len, prime_len, 
        quantize, steps_per_quarter, filter_4_4)
    
    metadata = get_metadata(dataset, genre_lookup)

    return pd.concat([metadata, embeddings], axis=1)


def get_embeddings(dataset, corpus, model, mem_len=512, prime_len=512, quantize=True, steps_per_quarter=4, filter_4_4=False):
    """
    For each data point in <data>, extract tokenised form and get
    hidden state embedding from <model>

    Param
    =====
    dataset: list
        List of datapoints from tf.datasets
    corpus: corpus
        Corpus with tokeniser
    model: model
        Trained transformer model (on corpus)
    mem_len: int
        Memory length of the model
    prime_len: int
        How many of the most recent tokens to prime model with before
        extracting embedding (if None use whole sequence)
    quantize: bool
        Quantize sequence before tokenisation?
    steps_per_quarter: int
        Step resolution of <quantize>
    filter_4_4: bool
        Filter to include only 4/4 time sigs?
    """
    dev_sequences = [mm.midi_to_note_sequence(d["midi"]) for d in dataset]

    if quantize:
        dev_sequences = [corpus._quantize(d, steps_per_quarter) for d in dev_sequences]

    if filter_4_4:
        dev_sequences = [s for s in dev_sequences if corpus._is_4_4(s)]

    tokens = [corpus._tokenize(d, steps_per_quarter, quantize) for d in dev_sequences]

    embeddings = [get_embedding(s, model, mem_len, prime_len) for s in tokens]
    return embeddings
    
    #num_features = embeddings[0].shape[0]
    #feat_names = [f'feat_{i}' for i in range(num_features)]
#
    #df = pd.DataFrame(embeddings, columns=feat_names)
    #
    #return df


def get_metadata(dataset, genre_lookup):
    """
    Return dataframe of metadata from <dataset>
    """
    primary_style = [genre_lookup[d['style']['primary']] for d in dataset]
    secondary_style = [d['style']['secondary'] for d in dataset]
    drummer = [d['style'] for d in dataset]
    return pd.DataFrame({
            'primary_style':primary_style,
            'secondary_style':secondary_style,
            'drummer':drummer})


# Load model
def get_embedding(seq, model, mem_len, prime_len=None):
    """
    Prime <model> with <seq> and return embedding
    
    Params
    ======
    seq: list
        List of tokens to prime with
    model: model
        Trained transformer xl model
    mem_len: int
        Mem_len of transformer
    prime_len: int
        If <prime_len>, prime with the most recent <prime_len> tokens
        else use all tokens
    """
    prime_len = prime_len if prime_len else len(seq) - 1

    sampler = TxlSimpleSampler(model, device, mem_len=mem_len)

    _, primed = prime_sampler(sampler, seq, prime_len)

    # Hidden states
    mems = sampler.mems

    return mems
## Final state
#final_state = mems[0].transpose(0,1)[0][-1]

#return final_state.numpy()


def learn_tsne(df, title='TSNE of training sample embeddings', label='primary_style'):
    """
    Train TSNE on embeddings in <df>
    Plot and annotate with style label
    """
    data = TSNE(n_components=2).fit_transform(df[[x for x in df.columns if 'feat' in x]].values)
    pd_data = pd.DataFrame(data, columns=['x','y'])
    pd_data['labels'] = df[label].values

    #rev_label = [lookup_genre[i] for i in pd_data['labels']]
    
    num_labels = len(set(pd_data['labels']))
    colours = random.choices(list(cnames.keys()), k=num_labels)

    fig = plt.figure(figsize=(8,8))

    for i,l in enumerate(pd_data['labels'].unique()):
        d = pd_data.loc[pd_data['labels'] == l]
        plt.scatter('x', 'y', data=d, color=colours[i], label=l)

    plt.title(title)
    plt.legend(loc="upper right")

    return plt

#tsne = learn_tsne(df)
#tsne.savefig('/Users/tom/Desktop/tsne.png')



def create_plot(X, y):
    pass