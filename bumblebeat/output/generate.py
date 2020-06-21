# Courtesy of chrisdonahue: https://github.com/kimiyoung/transformer-xl/issues/49
import torch
import torch.nn.functional as F
import numpy as np

import magenta.music as mm
from magenta.music.protobuf import music_pb2

from bumblebeat.transformer import MemTransformerLM
from bumblebeat.utils.data import split_range, create_dir_if_not_exists

"""
# To Use
    path = 'gpu_run-groove/full-midionly/20200620-222924/model.pt'
    USE_CUDA = False
    batch_size = 1
    tgt_len = 1
    ext_len = 0
    mem_len = 2000
    clamp_len = 1000
    gen_len = 2
    same_len = True

    device = torch.device("cuda" if USE_CUDA else "cpu")

    model = load_model(path, tgt_len, ext_len, mem_len, clamp_len, same_len, device)
    seq = generate_sequence(model, gen_len, batch_size, tgt_len, device)
"""

def load_model(path, tgt_len, ext_len, mem_len, clamp_len, same_len, device):
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

    # Change training length/memory attrs
    model.reset_length(tgt_len, ext_len, mem_len)
    if clamp_len > 0:
        model.clamp_len = clamp_len
    if same_len:
        model.same_len = True

    # Turn on evaluation mode which disables dropout.
    model.eval()

    return model


def generate_sequence(model, gen_len, batch_size, tgt_len, device):
    """
    Generate sample of len <gen_len> using pretrained transformer <model>
    """
    # Generate sequences of specified length and number
    with torch.no_grad():
        # Create buffer for generated sequences
        samples = torch.zeros([0, batch_size], dtype=torch.int64).to(device)

        # Initialize state
        prev_token = torch.zeros([tgt_len, batch_size], dtype=torch.int64).to(device)
        mems = tuple()

        # Autoregressive sampling
        for i in range(gen_len):
            ret = model.forward_generate(prev_token, *mems)

            # Retrieve logits and memory
            logits, mems = ret[0], ret[1:]

            # Ignore <S> (end of sequence) logit
            logits = logits[:, :, 1:]

            # Compute probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from probabilities
            sampler = torch.distributions.categorical.Categorical(probs=probs)
            token = sampler.sample()

            # Shift by one because we ignored <S> earlier
            token += 1

            # Add new token to buffer and update history
            samples = torch.cat([samples, token], dim=0)
            prev_token = token

    return samples


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
    special_token = max(pitch_vocab.keys()) + 1
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



