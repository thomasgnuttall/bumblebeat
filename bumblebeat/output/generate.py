# Courtesy of chrisdonahue: https://github.com/kimiyoung/transformer-xl/issues/49
import torch
import torch.nn.functional as F
import numpy as np

import magenta.music as mm
from magenta.music.protobuf import music_pb2

from bumblebeat.transformer import MemTransformerLM
from bumblebeat.utils.data import split_range, create_dir_if_not_exists
from bumblebeat.utils.generation import TxlSimpleSampler
"""
# To Use
    # lower temp and topk sampling


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
    seqs = generate_sequences(
                    model,
                    num=1, 
                    gen_len=gen_len, 
                    mem_len=mem_len, 
                    device=device, 
                    temp=0.80, 
                    topk=32)
    for i,s in enumerate(seqs):
        note_sequence = tokens_to_note_sequence(
            s[1:], 
            pitch_vocab, 
            simplified_pitches, 
            10, 
            time_vocab, 
            143.99988480009216)
        note_sequence_to_midi_file(note_sequence, f'/Users/tom/Desktop/bug_fix_2_{i}.midi')

"""

def load_model(path, device):
    """
    Load pretrained Transformer model for auto-regressive prediction
    """
    # Load the best saved model
    with open(path, 'rb') as f:
        model = torch.load(f, map_location=device)

    model.backward_compatible()
    model = model.to(device)

    # Make sure model uses vanilla softmax
    if model.sample_softmax > 0:
        raise NotImplementedError()
    if model.crit.n_clusters != 0:
        raise NotImplementedError()

    # Turn on evaluation mode which disables dropout.
    model.eval()

    return model


def generate_sequences(model, num, gen_len, mem_len, device, temp, topk=32):
    """
    Generate samples of len <gen_len> using pretrained transformer <model>
    """
    all_seqs = []
    # Generate sequences of specified length and number
    for i in range(num):
        sampler = TxlSimpleSampler(model, device, mem_len=mem_len)
        seq = [0]
        for _ in range(gen_len):
            token, _ = sampler.sample_next_token_updating_mem(
              seq[-1], temp=temp, topk=topk)
            seq.append(token)
        all_seqs.append(seq)

    return all_seqs


def accompany_sequence(model, seq, gen_len, temp, topk, mem_len, device):
    """
    Continue/accompany sequence, <seq> sampling from <model>

    Param
    =====
    model: 
        Trained transformer model
    seq: list
        Tokenised sequence to accompany
    gen_len: int
        How many tokens to generate
    temp: float
        Between 0 and 1. 
        1 samples from model prob dist
        0 always takes most likely
    topk: n
        k for topk sampling
    mem_len: int
        memory length of model
    device: torch device
        cpu or gpu

    Return
    ======
    Accompanying tokenised sequence (needs to be joined to original using join_sequences())

    """
    assert gen_len <= len(seq), "Cannot accompany beyond length of input sequence"
    sampler = TxlSimpleSampler(model, device, mem_len=memlen)

    inp = 0
    nll = 0.
    rhythm = []
    for i in range(gen_len):
        _, probs = sampler.sample_next_token_updating_mem(inp, exclude_eos=False)
        _probs = probs.cpu().numpy()
        
        _probmask = np.zeros_like(_probs)
        _probmask[seq] = 1.
        _probs *= _probmask
        
        if topk is not None:
            ind = np.argpartition(_probs, -topk)[-topk:]
            _probmask = np.zeros_like(_probs)
            _probmask[seq] = 1.
            _probs *= _probmask
        
        _probs /= np.sum(_probs)
        tar = np.random.choice(corpus.vocab_size, p=_probs)
        assert tar in seq
        rhythm.append(tar)

        inp = tar
    return rhythm


def continue_sequence(model, seq, prime_len, gen_len, temp, topk, mem_len, device):
    """
    Continue/accompany sequence, <seq> sampling from <model>

    Param
    =====
    model: 
        Trained transformer model
    seq: list
        Tokenised sequence to continue
    prime_len: int
        How many of thje most recent tokens in <seq> to 
        use to prime the model
    gen_len: int
        How many tokens to generate
    temp: float
        Between 0 and 1. 
        1 samples from model prob dist
        0 always takes most likely
    topk: n
        k for topk sampling
    mem_len: int
        memory length of model
    device: torch device
        cpu or gpu

    Return
    ======
    Original tokenised sequence continued by <gen_len> tokens

    """
    assert len(seq) >= prime_len + 1, "Insufficient tokens for prime length"

    sampler = TxlSimpleSampler(model, device, mem_len=mem_len)

    sampler = prime_sampler(sampler, seq, prime_len)

    nll = 0.
    cont = seq[:]
    for i in range(gen_len):
        gen, probs = sampler.sample_next_token_updating_mem(inp, temp=temp, topk=topk)
        p = probs[gen].cpu().item()
        nll += -np.log(p)
        inp = gen
        cont.append(gen)

    return cont


def prime_sampler(sampler, seq, prime_len):
    """
    Prime TXSimpleSampler with <seq> using <prime_len>
    """
    inp = 0
    nll = 0.
    for i in range(prime_len):
        tar = seq[i + 1]
        _, probs = sampler.sample_next_token_updating_mem(inp, exclude_eos=False)
        p = probs[tar].cpu().item()
        nll += -np.log(p)
        inp = tar

    return sampler
    

def tokens_to_note_sequence(tokens, pitch_vocab, pitch_classes, n_vel_buckets, time_vocab, qpm, time_sig=(4,4), ticks_per_quarter=480):
    """
    Convert sequence of tokens to note_sequence

    Param
    =====
    tokens: sequence
        Sequence of tokens to convert to note_sequence
    pitch_vocab: dict
        Dict of token:(pitch,velocity) 
    pitch_classes: list of lists
        list of lists indicating grouping of similar percussion instruments
        A random candidate will be taken from each group
    n_vel_buckets: int
        Number of velocity buckets
    time_vocab: dict
        token:number of silence ticks
    qpm: int
        quarters per minute
    time_sig: tuple
        time signature, (numerator, denominator)
    ticks_per_quarter: int
        Ticks per quarter

    Return
    ======
    music_pb2.NoteSequence
    """
    # Token to mark separation between samples
    time_tokens = list(time_vocab.values())
    reverse_time_vocab = {v:k for k,v in time_vocab.items()}

    ticks_per_second = ticks_per_quarter*qpm/60

    these_pitches = [np.random.choice(p) for p in pitch_classes]

    seq = music_pb2.NoteSequence()
    silence_ticks = 0
    for t in tokens:
        # Aggregate periods of silent ticks
        if t in time_tokens:
            silence_ticks += reverse_time_vocab[t]
        else:
            p, vel_bucket = pitch_vocab[t]
            pitch = these_pitches[p]

            vel = generate_velocity_in_bucket(vel_bucket, n_vel_buckets)

            start_time = silence_ticks/ticks_per_second
            end_time = start_time + 0.1 # TODO make this relative to qpm

            seq.notes.add(
                    pitch=pitch,
                    velocity=vel,
                    start_time=start_time,
                    end_time=end_time,
                    is_drum=True
                )

    seq.ticks_per_quarter = ticks_per_quarter
    seq.tempos.add(qpm=qpm)
    seq.time_signatures.add(numerator=time_sig[0], denominator=time_sig[1])

    return seq


def generate_velocity_in_bucket(bucket, n_buckets):
    """
    Generate a random velocity in <bucket> for range of <n_buckets>
        (0 -> 127 possible)
    """
    srange = split_range(0, 127, n_buckets)

    low = srange[bucket]
    high = srange[bucket+1]

    vel = np.random.uniform(low=low, high=high)
    return int(vel)


def note_sequence_to_midi_file(note_sequence, path):
    """
    Save <note_sequence> to .midi file at <path>
    """
    create_dir_if_not_exists(path)
    mm.sequence_proto_to_midi_file(note_sequence, path)


def note_sequence_to_audio_file(note_sequence, path):
    """
    Save <note_sequence> to .wav file at <path>
    """
    pass



