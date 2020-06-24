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

original = dev_sequences[2]
tokens = corpus._tokenize(original)
reconstructed = tokens_to_note_sequence(
							tokens, 
							corpus.reverse_vocab, 
							simplified_pitches, 
							10, 
							time_steps_vocab, 
							143.99988480009216,
							time_sig=(4,4), 
							ticks_per_quarter=original.ticks_per_quarter)


#_tokenize
triples = [(corpus.pitch_class_map[n.pitch], \
      get_bucket_number(n.velocity, corpus.velocity_buckets),
      n.start_time) \
        for n in original.notes \
        if n.pitch in corpus.pitch_class_map]

ticks_per_quarter = 480
qpm = original.tempos[0].qpm # quarters per minute
ticks_per_second = qpm*ticks_per_quarter/60

pitch_vocab = corpus.vocab

# Initalise final tokenised sequence
w_silence = []

# Initalise counter to keep track of consecutive pitches
# so that we can ensure they are appended to our
# final tokenised sequence in numerical order
consecutive_pitches = 0

# index, (pitch, velocity, start time)
for i, (x, y, z) in enumerate(triples):
    if i == 0:
        silence = z
    else:
        silence = z - triples[i-1][2] # z of previous element
        import ipdb; ipdb.set_trace()
    ticks = int(silence*ticks_per_second)
    if ticks > ticks_per:
        # make sure that any consecutive pitches in sequence
        # are in numerical order so as to enforce an ordering
        # rule for pitches that are commonly hit in unison
        w_silence[-consecutive_pitches:] = sorted(w_silence[-consecutive_pitches:])

        # Since silences are computed using time since last pitch class,
        # every iteration in this loop is a pitch class.
        # Hence we set consecutive pitch back to one 
        # (representing the pitch of this iteration, added just outside of this if-clause)
        consecutive_pitches = 1

        # Number of ticks to list of time tokens
        time_tokens = corpus._convert_num_to_denominations(ticks, time_steps_vocab)

        # Add time tokens to final sequence before we add our pitch class
        w_silence += time_tokens
    else:
        # Remember that every iteration is a pitch.
        # If <ticks> is 0 then this pitch occurs
        # simultaneously with the previous.
        # We sort these numerically before adding the
        # next stream of time tokens
        consecutive_pitches += 1

    # Triple to tokens...
    #   Discard time since we have handled that with time tokens.
    #   Look up pitch velocity combination for corresponding token.
    pitch_tok = pitch_vocab[x][y] # [pitch class][velocity]
    w_silence.append(pitch_tok)








# Remove and fill list with silence vocab
if quantize:
    #total_steps = note_sequence.total_quantized_steps
    #timestep_lim = corpus._roundup(total_steps, 4)
    #
    #filled = fill_timestep_silence(d, timestep_lim, corpus.time_steps_vocab)
#else:
    ticks_per_quarter = note_sequence.ticks_per_quarter
    qpm = note_sequence.tempos[0].qpm # quarters per minute
    ticks_per_second = qpm*ticks_per_quarter/60

    filled = corpus._tokenize_w_ticks(d, ticks_per_second, corpus.vocab, time_steps_vocab)