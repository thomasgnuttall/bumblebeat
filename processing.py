%load_ext autoreload

%autoreload 2

# very important: https://www.twilio.com/blog/training-a-neural-network-on-midi-music-data-with-magenta-and-python

import tensorflow_datasets as tfds
import tensorflow as tf

from magenta import music as mm

print('Installing dependencies...')

#!apt-get update -qq && apt-get install -qq libfluidsynth1 fluid-soundfont-gm build-essential libasound2-dev libjack-dev
#!pip install -q pyfluidsynth
#!pip install -U -q magenta

import tensorflow_datasets as tfds
import tensorflow as tf

# Allow python to pick up the newly-installed fluidsynth lib.
# This is only needed for the hosted Colab environment.
import ctypes.util
orig_ctypes_util_find_library = ctypes.util.find_library
def proxy_find_library(lib):
  if lib == 'fluidsynth':
    return 'libfluidsynth.so.1'
  else:
    return orig_ctypes_util_find_library(lib)
ctypes.util.find_library = proxy_find_library
  
print('Importing software libraries...')

import copy, warnings, librosa, numpy as np
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Colab/Notebook specific stuff
import IPython.display
from IPython.display import Audio
#from google.colab import files

# Magenta specific stuff
from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel
from magenta import music as mm
from magenta.music import midi_synth
from magenta.music.sequences_lib import concatenate_sequences
from magenta.models.music_vae import data
#from magenta.protobuf import music_pb2

# Define some functions

# If a sequence has notes at time before 0.0, scootch them up to 0
def start_notes_at_0(s):
  for n in s.notes:
    if n.start_time < 0:
      n.end_time -= n.start_time
      n.start_time = 0
  return s

# Some midi files come by default from different instrument channels
# Quick and dirty way to set midi files to be recognized as drums
def set_to_drums(ns):
  for n in ns.notes:
    n.instrument=9
    n.is_drum = True
    
def unset_to_drums(ns):
  for note in ns.notes:
    note.is_drum=False
    note.instrument=0
  return ns

# quickly change the tempo of a midi sequence and adjust all NoteSequence
def change_tempo(note_sequence, new_tempo):
  new_sequence = copy.deepcopy(note_sequence)
  ratio = note_sequence.tempos[0].qpm / new_tempo
  for note in new_sequence.notes:
    note.start_time = note.start_time * ratio
    note.end_time = note.end_time * ratio
  new_sequence.tempos[0].qpm = new_tempo
  return new_sequence

def download(note_sequence, filename):
  mm.sequence_proto_to_midi_file(note_sequence, filename)
  files.download(filename)

# Load some configs to be used later
dc_quantize = configs.CONFIG_MAP['groovae_2bar_humanize'].data_converter
dc_tap = configs.CONFIG_MAP['groovae_2bar_tap_fixed_velocity'].data_converter
dc_hihat = configs.CONFIG_MAP['groovae_2bar_add_closed_hh'].data_converter
dc_4bar = configs.CONFIG_MAP['groovae_4bar'].data_converter

# quick method for removing microtiming and velocity from a sequence
def get_quantized_2bar(s, velocity=0):
  new_s = dc_quantize.to_notesequences(dc_quantize.to_tensors(s).inputs)[0]
  new_s = change_tempo(new_s, s.tempos[0].qpm)
  if velocity != 0:
    for n in new_s.notes:
      n.velocity = velocity
  return new_s

# quick method for turning a drumbeat into a tapped rhythm
def get_tapped_2bar(s, velocity=85, ride=False):
  new_s = dc_tap.to_notesequences(dc_tap.to_tensors(s).inputs)[0]
  new_s = change_tempo(new_s, s.tempos[0].qpm)
  if velocity != 0:
    for n in new_s.notes:
      n.velocity = velocity
  if ride:
    for n in new_s.notes:
      n.pitch = 42
  return new_s

# quick method for removing hi-hats from a sequence
def get_hh_2bar(s):
  new_s = dc_hihat.to_notesequences(dc_hihat.to_tensors(s).inputs)[0]
  new_s = change_tempo(new_s, s.tempos[0].qpm)
  return new_s

# Calculate quantization steps but do not remove microtiming
def quantize(s, steps_per_quarter=4):
  return mm.sequences_lib.quantize_note_sequence(s,steps_per_quarter)

# Destructively quantize a midi sequence
def flatten_quantization(s):
  beat_length = 60. / s.tempos[0].qpm
  step_length = beat_length / 4 #s.quantization_info.steps_per_quarter
  new_s = copy.deepcopy(s)
  for note in new_s.notes:
    note.start_time = step_length * note.quantized_start_step
    note.end_time = step_length * note.quantized_end_step
  return new_s

# Calculate how far off the beat a note is
def get_offset(s, note_index):
  q_s = flatten_quantization(quantize(s))
  true_onset = s.notes[note_index].start_time
  quantized_onset = q_s.notes[note_index].start_time
  diff = quantized_onset - true_onset
  beat_length = 60. / s.tempos[0].qpm
  step_length = beat_length / 4 #q_s.quantization_info.steps_per_quarter
  offset = diff/step_length
  return offset

def is_4_4(s):
  ts = s.time_signatures[0]
  return (ts.numerator == 4 and ts.denominator == 4)

def preprocess_4bar(s):
  return dc_4bar.to_notesequences(dc_4bar.to_tensors(s).outputs)[0]

def preprocess_2bar(s):
  return dc_quantize.to_notesequences(dc_quantize.to_tensors(s).outputs)[0]

def _slerp(p0, p1, t):
  """Spherical linear interpolation."""
  omega = np.arccos(np.dot(np.squeeze(p0/np.linalg.norm(p0)),
    np.squeeze(p1/np.linalg.norm(p1))))
  so = np.sin(omega)
  return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1


print("Download MIDI data...")

# Load MIDI files from GMD with MIDI only (no audio) as a tf.data.Dataset
dataset_2bar = tfds.as_numpy(tfds.load(
    name="groove/2bar-midionly",
    split=tfds.Split.VALIDATION,
    try_gcs=True))

# Convert midi string to midi format, quantize notes
dev_sequences = [quantize(mm.midi_to_note_sequence(features["midi"])) for features in dataset_2bar]

# Filter out those that are not in 4/4 and do not have any notes
dev_sequences = [s for s in dev_sequences if is_4_4(s) and len(s.notes) > 0 and s.notes[-1].quantized_end_step > mm.steps_per_bar_in_quantized_sequence(s)]

from magenta.models.music_vae import data

grooveConverter = data.GrooveConverter(max_tensors_per_notesequence=20)

tensors = [grooveConverter._to_tensors(s) for s in dev_sequences]

inputs = [t[0][0] for t in tensors]
input_dataset = tf.data.Dataset.from_tensor_slices(inputs[:5])

outputs =[t[1][0] for t in tensors]

input_dataset = tf.data.Dataset.from_tensor_slices(inputs)
output_dataset = tf.data.Dataset.from_tensor_slices(outputs)

## Polyphonic Entrypoint
#   # Copyright 2020 The Magenta Authors.
#   #
#   # Licensed under the Apache License, Version 2.0 (the "License");
#   # you may not use this file except in compliance with the License.
#   # You may obtain a copy of the License at
#   #
#   #     http://www.apache.org/licenses/LICENSE-2.0
#   #
#   # Unless required by applicable law or agreed to in writing, software
#   # distributed under the License is distributed on an "AS IS" BASIS,
#   # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   # See the License for the specific language governing permissions and
#   # limitations under the License.
#   
#   """Create a dataset of SequenceExamples from NoteSequence protos.
#   
#   This script will extract polyphonic tracks from NoteSequence protos and save
#   them to TensorFlow's SequenceExample protos for input to the polyphonic RNN
#   models.
#   """
#   
#   import os
#   
#   from magenta.models.polyphony_rnn import polyphony_model
#   from magenta.models.polyphony_rnn import polyphony_rnn_pipeline
#   from magenta.pipelines import pipeline
#   import tensorflow.compat.v1 as tf
#   
#   flags = tf.app.flags
#   FLAGS = tf.app.flags.FLAGS
#   flags.DEFINE_string(
#       'input', None,
#       'TFRecord to read NoteSequence protos from.')
#   flags.DEFINE_string(
#       'output_dir', None,
#       'Directory to write training and eval TFRecord files. The TFRecord files '
#       'are populated with SequenceExample protos.')
#   flags.DEFINE_float(
#       'eval_ratio', 0.1,
#       'Fraction of input to set aside for eval set. Partition is randomly '
#       'selected.')
#   flags.DEFINE_string(
#       'log', 'INFO',
#       'The threshold for what messages will be logged DEBUG, INFO, WARN, ERROR, '
#       'or FATAL.')
#   
#   
#   def main(unused_argv):
#     tf.logging.set_verbosity(FLAGS.log)
#   
#     pipeline_instance = polyphony_rnn_pipeline.get_pipeline(
#         min_steps=80,  # 5 measures
#         max_steps=512,
#         eval_ratio=FLAGS.eval_ratio,
#         config=polyphony_model.default_configs['polyphony'])
#   
#     input_dir = os.path.expanduser(FLAGS.input)
#     output_dir = os.path.expanduser(FLAGS.output_dir)
#     pipeline.run_pipeline_serial(
#         pipeline_instance,
#         pipeline.tf_record_iterator(input_dir, pipeline_instance.input_type),
#         output_dir)
#   
#   
#   def console_entry_point():
#     tf.app.run(main)
#   
#   
#   if __name__ == '__main__':
#     console_entry_point()

####### DAG creation for polyphony
#   transposition_range = range(-4, 5)

#   partitioner = pipelines_common.RandomPartition(
#       music_pb2.NoteSequence,
#       ['eval_poly_tracks', 'training_poly_tracks'],
#       [eval_ratio])
#   dag = {partitioner: dag_pipeline.DagInput(music_pb2.NoteSequence)}

#   for mode in ['eval', 'training']:
#     time_change_splitter = note_sequence_pipelines.TimeChangeSplitter(
#         name='TimeChangeSplitter_' + mode)
#     quantizer = note_sequence_pipelines.Quantizer(
#         steps_per_quarter=config.steps_per_quarter, name='Quantizer_' + mode)
#     transposition_pipeline = note_sequence_pipelines.TranspositionPipeline(
#         transposition_range, name='TranspositionPipeline_' + mode)
#     poly_extractor = PolyphonicSequenceExtractor(
#         min_steps=min_steps, max_steps=max_steps, name='PolyExtractor_' + mode)
#     encoder_pipeline = event_sequence_pipeline.EncoderPipeline(
#         polyphony_lib.PolyphonicSequence, config.encoder_decoder,
#         name='EncoderPipeline_' + mode)

#     dag[time_change_splitter] = partitioner[mode + '_poly_tracks']
#     dag[quantizer] = time_change_splitter
#     dag[transposition_pipeline] = quantizer
#     dag[poly_extractor] = transposition_pipeline
#     dag[encoder_pipeline] = poly_extractor
#     dag[dag_pipeline.DagOutput(mode + '_poly_tracks')] = encoder_pipeline

### THAT TIME I WAS CLOSE

##   #### MAYBE #####
##   # Note sequence -> polyphony
##   from magenta.models.polyphony_rnn import polyphony_rnn_pipeline

##   poly_sequence_extractor = polyphony_rnn_pipeline.PolyphonicSequenceExtractor(min_steps=0, max_steps=512)

##   sequence_examples = [poly_sequence_extractor.transform(i) for i in dev_sequences]

##   # polyphonic sequence -> model encoding
##   # in music_vae.data there are full drum pitch classes (ROLAND)

##   ROLAND_DRUM_PITCH_CLASSES = [
##       # kick drum
##       [36],
##       # snare drum
##       [38, 37, 40],
##       # closed hi-hat
##       [42, 22, 44],
##       # open hi-hat
##       [46, 26],
##       # low tom
##       [43, 58],
##       # mid tom
##       [47, 45],
##       # high tom
##       [50, 48],
##       # crash cymbal
##       [49, 52, 55, 57],
##       # ride cymbal
##       [51, 53, 59]
##   ]

##   from magenta.pipelines import event_sequence_pipeline
##   from magenta.models.polyphony_rnn import polyphony_lib

##   from magenta.models.polyphony_rnn import polyphony_encoder_decoder

##   from tensorflow.contrib import training as contrib_training

##   import magenta
##   from magenta.music import drums_encoder_decoder

##   encoder = magenta.music.OneHotEventSequenceEncoderDecoder(
##     drums_encoder_decoder.MultiDrumOneHotEncoding(drum_type_pitches=ROLAND_DRUM_PITCH_CLASSES, ignore_unknown_drums=False)
##   )

##   encoder_pipeline = event_sequence_pipeline.EncoderPipeline(
##       polyphony_lib.PolyphonicSequence, encoder)

##   encoded = [encoder_pipeline.transform(i) for i in sequence_examples]








