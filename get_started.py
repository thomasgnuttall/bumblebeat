# Load data (return class)


# Data representation reference
#Ian Simon and Sageev Oore. Performance RNN:
#Generating music with expressive timing and dynamics. https://magenta.tensorflow.org/
#performance-rnn, 2017.


# Preprocess and Create TF Records
# ================================
# (Get raw data into format ready for transformer)
# 
# - Different depending on CPU, GPU or TPU
# - Wrap in pipeline ready Corpus class
#   - Identify vocabulary (class of its own) 
#       - Count and identify unique "tokens"
#       - ¿Vocabs consistent across datasets?
#       - Important to consider encoding of "special" tokens
#       - Attribute of the corpus
#   - Store vocab as file associated with dataset (for future)
#   - Load vocab
#   - Create dictionary of index to symbol and vice versa (as attribute of vocab)
#   - Consider filtering out uncommon tokens (set max_size of vocab)
#   - Load and encode data into numpy array (replace tokens with index)
#       - For train, test and validation (each an attribute of the corpus)
#   - Pickle corpus
#   - Pickle alongside metadata of corpus in dictionary
# - Seperately store train,test, validation to tensorflow records
#   - With accompanying metadata json
#   - Binary storage format of tensorflow
#       - useful for streaming data over a network
#       - useful for aggregating datasets
#       - integrates with TF nicely
#       - datasets arent stored in RAM
#       - Very efficient data import for sequence data


##################### Confifuration ########################
# - to be abstracted to configuration files at some point
conf = {
    'dataset_name': "groove/2bar-midionly",
    'data_dir': 'data/',
    'per_host_test_bsz': 4,
    'batch_size': 24,
    'tgt_len': 512, # number of steps to predict
    'num_core_per_host': 8, 
    'quantize': True, # After this line for groove converter
    'steps_per_quarter': 4,
    'filter_4_4': None, # maybe we dont want this?,
    'max_tensors_per_notesequence': 20
}

# Default list of 9 drum types, where each type is represented by a list of
# MIDI pitches for drum sounds belonging to that type. This default list
# attempts to map all GM1 and GM2 drums onto a much smaller standard drum kit
# based on drum sound and function.
DEFAULT_DRUM_TYPE_PITCHES = [
    # kick drum
    [36, 35],

    # snare drum
    [38, 27, 28, 31, 32, 33, 34, 37, 39, 40, 56, 65, 66, 75, 85],

    # closed hi-hat
    [42, 44, 54, 68, 69, 70, 71, 73, 78, 80, 22],

    # open hi-hat
    [46, 67, 72, 74, 79, 81, 26],

    # low tom
    [45, 29, 41, 43, 61, 64, 84],

    # mid tom
    [48, 47, 60, 63, 77, 86, 87],

    # high tom
    [50, 30, 62, 76, 83],

    # crash cymbal
    [49, 52, 55, 57, 58],

    # ride cymbal
    [51, 53, 59, 82]
]

# Any length silence of length shorter
# than 100,000 timesteps can be represented
# efficiently
# length of silence: token
time_steps_vocab = [
        1: 0,       
        10: 1,      
        100: 2,
        1000: 3,
        10000: 4
    ]
### Data Representation ####
############################

# dev_sequence:
#     notes length = 15
#         - pitch
#         - velocity
#         - start_time
#         - end_time
#         - is_drum
#         - quantized_start_step
#         - quantized_end_step

#     min(quantized_end_step) = 1
#     max(quantized_end_step) = 17
#     all pitches = {22, 26, 36, 38, 42, 46, 55} (n=7) (out of a possible 22)
#     unique velocities = {42, 53, 55, 68, 71, 72, 73, 80, 83, 95, 120, 127}

#     34 note attributes
#     1 ControlChange attribute (<class 'music_pb2.ControlChange'>)
#     1 descriptor attribute (<google.protobuf.pyext._message.MessageDescriptor object>)

# 
# ** Apply GrooveConverter **
#     - triples representation: (hit, velocity, offset)
#     - each timestep a fixed beat on a grid
#         - default spacing 16th notes
#     - binary hits [0, 1]
#     - velocities continuous between 0 and 1
#     - offsets continuous between -0.5 and 0.5 --rescaled--> -1 and 1 for tensors
#     - each timestep contains this representation for each of a fixed list of 9 categories (defined in drums_encoder_decoder.py) (can be changed)
#     - Each category has a triple representation (hit, velocity, offset) x 9 = 27
#     - One measure has 16 timesteps (16, 27)
#     - Dropout can be done here
#
# tensors:

# 4 x 1 x 32 x 27*
# [inputs, outputs, controls, lengths] x ? x measures x category_triple
# [hits_vectors, velocity_vectors, offset_vectors]


# - Experiment with different drum categories
# - Dropout in GrooveConverter
# - very important: https://www.twilio.com/blog/training-a-neural-network-on-midi-music-data-with-magenta-and-python
# - what the fuck are controls


# i = 26      # 4 x 1 x 32 x 27*

# ex_dev_sequence = dev_sequences[i]
# ex_tensor = tensors[i]

# # Each row a timestep
# # Each column corresponds to hit, velocity, offset for each drum cat
# # These are equal with no dropout etc...
# # these are lists of length one
# inputs = ex_tensor[0][0]
# outputs = ex_tensor[1][0]

# # These are irrelevant
# controls = ex_tensor[2] # empty in this case
# lengths = ex_tensor[3] # just one legnth in this case

# - Use a word 2 vec pretrained model to vectorise the inputs
# - Ignore embedding layer, use entire sequence as input to every time step
# - word2vec
# - ignore velocities/offset
# - last activation layer should be changed for offsets and velocities


# # learning to groove
    # - softmax on hits
    # - velocity to sigmoid
    # - offset to tanh

# 
# Groove has 22 instruments (separated into 9 categories)
#
# Plan
#   - start with just 9 instruments (no velocity or offeset)
#        - simplified mapping in groove midi paper (see groove midi page)
#   - experiment with 22 instruments
#   - experiment with adding velocity
#        - bucket into N
#   - experiment with adding offset
#       - two aproaches:
#           - include time tokens (like in Lakhnes)
#           - include offset buckeets with velocity;
#                - 9 instruments= 900 tokens, 22 = 22000 tokens
#   - experiment with adding time tokens (like LakhNes)
#
#
# With 10 offset buckets and 10 velocity buckets
# 

ex_tensor = tensors[0][0][0]

ex_timestep = ex_tensor[2]
# -> length = 9 x 3 = 27 ([9 instrument, 9 velocity, 9 offset])


encode_timestep(ex_timestep, token_dict, vel_buckets)

def split_range(r1, r2, n):
    """
    Split range <r1> - <r2> into <n> equal size buckets
    """
    step = (r2 - r1)/n
    return [r1+step*i for i in range(n+1)]


def get_bucket_number(value, srange):
    """
    Return index of bucket that <value> falls into

    srange is a list of bucket divisions from split_range()
    """
    assert srange == (sorted(srange)),\
        "srange must be sorted list"
    assert len(set(srange)) == len(srange),\
        "srange buckets must be unique"
    assert value <= max(srange) and value >= min(srange),\
        "value is not in any srange bucket"

    for i in range(len(srange)-1):
        if value <= srange[i+1]:
            return i

import itertools


def create_vocab(n_instruments, n_velocity_buckets, first_index=5):
    """
    Create vocabulary of all possible instrument-velocity combinations.

    <first_index> dictates which index to start on, default 5 to allow for
    5 timestep tokens.

    Each instrument is represented by <n_velocity_buckets> integers
    Tokens increase across the dimension of instruments first:
        token <first_index>     is instrument 0, velocity bucket 0
        token <first_index> + 1 is instrument 0, velocity bucket 1
        token <first_index> + 2 is instrument 0, velocity bucket 2
        token <first_index> + N is instrument 0, velocity bucket 3
            - where i = <n_instruments> - 1 (because of 0 indexing)
            - where j = <n_velocity_buckets> - 1 (because of 0 indexing)
            - where N = ixj - 1

    returns: 2 x dict
        {instrument_index: {velocity_index:token}}
            ...for all instruments, velocities and tokens
        {index: (instrument index, velocity index)},
    """
    
    # itertools.product returns sorted how we desire    
    all_comb = itertools.product(range(n_instruments), range(n_velocity_buckets))
    d_reverse = {i+first_index:(x,y) for i,(x,y) in enumerate(all_comb)}
    
    d = {i:{} for i in range(n_instruments)}
    for t, (i,v) in  d_reverse.items():
        d[i][v] = t

    return d, d_reverse


def encode_timestep(arr, token_dict, vel_range, n_instruments=None):
    """
    For one time step return array of tokens describing timestep

    Instruments are always returned in order of instrument index, velocity index

    Param
    =====
    arr: np.array
        array of timestep info, <n_instruments>*[hit event] + <n_intruments>*[velocity] + <n_instruments>*[offset] 
    token_dict: dict
        {instrument_index: {velocity_index:token}}
            ...for all instruments, velocities and tokens
        from create_vocab()
    vel_range: list
        list of velocity bucket boundaries 
        from split_range()
    n_instruments: int
        number of instruments described in arr
            if None, infer from token_dict
    """
    
    n = n_instruments or len(token_dict)

    indices = range(n)
    instruments = arr[:n]
    velocities = arr[n:n*2]
    offsets = arr[n*2:]

    # instrument, velocity for non 0 instruments
    hits = [(x,z) for x,y,z in zip(indices, instruments, velocities) if y != 0 and z != 0]

    # convert velocity to bucket
    hits_vel = [(x, get_bucket_number(z, vel_range)) for x,z in hits]

    # Get token corresponding to this instrument at this velocity
    tokens = [token_dict[i][v] for i,v in hits_vel]

    return np.array(tokens)



### transformer-xl/data_utils.py
################################
import math
import os
import pickle

import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.gfile import Exists as exists
from tensorflow.gfile import MakeDirs as makedirs
from tensorflow.gfile import Glob as glob

from magenta import music as mm
from magenta.models.music_vae import data as vae_data

def create_dir_if_not_exists(path):
    """
    If the directory at <path> does not exist, create it empty
    """
    directory = os.path.dirname(path)
    # Do not try and create directory if path is just a filename
    if (not os.path.exists(directory)) and (directory != ''):
        os.makedirs(directory)

def main(unused_argv):
    
    del unused_argv  # Unused

    # ARGS in future
    dataset_name = conf['dataset_name']
    data_dir = conf['data_dir']

    # ARGS for TF records
    per_host_test_bsz = conf['per_host_test_bsz']
    batch_size = conf['batch_size']
    tgt_len = conf['tgt_len'] # number of steps to predict
    num_core_per_host = conf['num_core_per_host']

    corpus = get_corpus(dataset_name, data_dir)

    save_dir = os.path.join(data_dir, "tfrecords")
    create_dir_if_not_exists(save_dir)

    # test mode
    if per_host_test_bsz > 0:
        corpus.convert_to_tfrecords(
          "valid", save_dir, per_host_test_bsz, tgt_len, num_core_per_host, conf
        )
        return

    #for split, batch_size in zip(
    #  ["train", "valid"],
    #  [FLAGS.per_host_train_bsz, FLAGS.per_host_valid_bsz]):

    #    if batch_size <= 0: continue
    #    print("Converting {} set...".format(split))
    #    corpus.convert_to_tfrecords(split, save_dir, batch_size, FLAGS.tgt_len,
    #                                FLAGS.num_core_per_host, FLAGS=FLAGS)


def get_corpus(dataset_name, data_dir):
    """
    Load groove data into custom Corpus class

    Returns
    =======
    bumblebeat.data.Corpus object

    """
    fn = os.path.join(data_dir, dataset_name, "cache.pkl")

    if exists(fn):
        print("Loading cached dataset...")
        with open(fn, "rb") as fp:
            corpus = pickle.load(fp)
    else:
        create_dir_if_not_exists(fn)

        print("Producing dataset...")
        corpus = Corpus(data_dir, dataset_name)
    
       #print("Saving dataset...")
       #with open(fn, "wb") as fp:
       #    pickle.dump(corpus, fp, protocol=2)

       #corpus_info = {
       #  "vocab_size" : len(corpus.vocab),
#      #   "cutoffs" : corpus.cutoffs,
       #  "dataset" : corpus.dataset_name
       #}
       #with open(os.path.join(data_dir, dataset_name, "corpus-info.json"), "w") as fp:
       #    json.dump(corpus_info, fp)

    return corpus


class Corpus:
    """
    Corpus to handle data in pipeline
    """
    def __init__(
            self, 
            data_dir, 
            dataset_name, 
            pitch_classes=DEFAULT_DRUM_TYPE_PITCHES,  
            time_steps_vocab=time_steps_vocab,
            n_velocity_buckets=10,
            min_velocity=0,
            max_velocity=127
        ):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.pitch_classes = pitch_classes
        self.time_steps_vocab = time_steps_vocab
        self.n_velocity_buckets = n_velocity_buckets

        self.velocity_buckets = split_range(
            min_velocity, max_velocity, n_velocity_buckets)
        self.pitch_class_map = self._classes_to_map(self.pitch_classes)
        self.n_instruments = len(set(self.pitch_class_map.values()))

        print(f'Generating vocab of {self.n_instruments} instruments and {n_velocity_buckets} velocity buckets')
        self.vocab, self.reverse_vocab = create_vocab(
                        self.n_instruments, n_velocity_buckets, 
                        first_index=len(time_steps_vocab)
                    ) # leave initial indices for time steps vocab

        #self.train_data = self.download_midi(dataset_name, tfds.Split.TRAIN)
        #self.test_data = self.download_midi(dataset_name, tfds.Split.TEST)
        valid_data = self.download_midi(dataset_name, tfds.Split.VALIDATION)
        #self.all_data = self.download_midi(dataset_name, tfds.Split.ALL)

        self.valid = self.process_dataset(valid_data)

    def download_midi(self, dataset_name, dataset_split):
        print(f"Downloading midi data: {dataset_name}, split: {dataset_split}")
        dataset = tfds.as_numpy(
            tfds.load(
                name=dataset_name,
                split=dataset_split,
                try_gcs=True
            ))
        return dataset

    def process_dataset(self, dataset):
        """
        Create tensors of triple representation for each sample
        (hit, velocity, offset) for each midi instrument at each timestep
        """
        print('Processing dataset...')
        # Abstract to ARGS at some point
        quantize = conf['quantize']
        steps_per_quarter  = conf['steps_per_quarter']
        filter_4_4 = conf['filter_4_4'] # maybe we dont want this?
        max_tensors_per_notesequence = conf['max_tensors_per_notesequence']

        # To midi note sequence using magent
        dev_sequences = [mm.midi_to_note_sequence(features["midi"]) for features in dataset]
        
        if quantize:
            dev_sequences = [self._quantize(d, steps_per_quarter) for d in dev_sequences]

        # Filter out those that are not in 4/4 and do not have any notes
        dev_sequences = [
            s for s in dev_sequences 
            if self._is_4_4(s) and len(s.notes) > 0
            and s.notes[-1].quantized_end_step > mm.steps_per_bar_in_quantized_sequence(s)
        ]

        # note sequence -> [(pitch, vel_bucket, start timestep)]
        triples = [self._note_sequence_to_triple(d) for d in dev_sequences]

        # Remove (,start timestep) and fill list with silence vocab
        pitch_vel_timestep = [fill_timestep_silence(t, self.time_steps_vocab) \
                              for t in triples]
        return triples

        ## Create triple representation
        ## Loads of ARGS to pass here
        #grooveConverter = vae_data.GrooveConverter(
        #    max_tensors_per_notesequence=max_tensors_per_notesequence)
        #
        #self.pitch_class_map = grooveConverter.pitch_class_map
        #self.vocab = set(self.pitch_class_map.values())
        #self.dev_sequences = dev_sequences

        #return [grooveConverter._to_tensors(s) for s in dev_sequences]

    def convert_to_tf_records(self, split, save_dir, bsz, tgt_len,
                        num_core_per_host, conf):
        """
        Convert tensor data to TF records and store
        """
        file_names = []
        record_name = f"record_info-{split}.bsz-{bsz}.tlen-{tgt_len}.json"
        record_info_path = os.path.join(save_dir, record_name)

        data = getattr(self, split)
        
        file_name, num_batch = create_ordered_tfrecords(
            save_dir, split, data, bsz, tgt_len, num_core_per_host,
            [], [],
            num_passes=FLAGS.num_passes if split == 'train' and use_tpu else 1,
            use_tpu=use_tpu)

        file_names.append(file_name)

        with open(record_info_path, "w") as fp:
            record_info = {
              "filenames": file_names,
              "bin_sizes": bin_sizes,
              "num_batch": num_batch
            }
            json.dump(record_info, fp)

    def _quantize(self, s, steps_per_quarter=4):
        """
        Quantize a magenta Note Sequence object
        """
        return mm.sequences_lib.quantize_note_sequence(s,steps_per_quarter)

    def _is_4_4(self, s):
        """
        Return True if sample, <s> is in 4/4 timing, False otherwise
        """
        ts = s.time_signatures[0]
        return (ts.numerator == 4 and ts.denominator == 4)

    def _note_sequence_to_triple(self, note_sequence):
        """
        from magenta <note_sequence> return list of
        (pitch, velocity, quantized_start_step) 
            we dont care about quantized_end_step 
            since we are dealing with drums

        - pitch is mapped using self.pitch_class_map
        - velocities are bucketted as per self.velocity_buckets
        """
        d = [(self.pitch_class_map[n.pitch], \
              get_bucket_number(n.velocity, self.velocity_buckets),
              n.quantized_start_step) \
                for n in note_sequence.notes \
                if n.pitch in self.pitch_class_map]
        return d

    def _classes_to_map(self, classes):
        class_map = {}
        for cls, pitches in enumerate(classes):
            for pitch in pitches:
                class_map[pitch] = cls
        return class_map







def create_ordered_tfrecords(save_dir, basename, data, batch_size, tgt_len,
                               num_core_per_host, cutoffs=[], bin_sizes=[], 
                               num_passes=1):

    file_name = f"{basename}.bsz-{batch_size}.tlen-{tgt_len}.tfrecords"

    save_path = os.path.join(save_dir, file_name)
    record_writer = tf.python_io.TFRecordWriter(save_path)

    batched_data = batchify(data, batch_size, num_passes)

    num_batch = 0
    # for t in range(0, batched_data.shape[1] - tgt_len - 1, tgt_len):
    for t in range(0, batched_data.shape[1] - 1, tgt_len):
      cur_tgt_len = min(batched_data.shape[1] - 1 - t, tgt_len)
      # drop the remainder if use tpu
      if use_tpu and cur_tgt_len < tgt_len: 
        break
      if num_batch % 500 == 0:
        print("  processing batch {}".format(num_batch))
      for idx in range(batch_size):
        inputs = batched_data[idx, t:t + cur_tgt_len]
        labels = batched_data[idx, t + 1:t + cur_tgt_len + 1]

        # features dict
        feature = {
            "inputs": _int64_feature(inputs),
            "labels": _int64_feature(labels),
        }

        if len(cutoffs) > 0 and use_tpu:
          # validate `bin_sizes` and `cutoffs`
          assert len(cutoffs) - len(bin_sizes) == 2, \
            "len(cutoffs) - len(bin_sizes) != 2"

          # mask for bin 0
          left, right = cutoffs[:2]
          inp_mask = ((inputs >= left) * (inputs < right)).astype(np.float32)
          tgt_mask = ((labels >= left) * (labels < right)).astype(np.float32)

          feature["inp_mask"] = _float_feature(inp_mask)
          feature["tgt_mask"] = _float_feature(tgt_mask)

          # refresh `inp_cnts` and `tgt_cnts` for each TPU core
          if idx % (batch_size // num_core_per_host) == 0:
            inp_cnts = [0] * len(bin_sizes)
            tgt_cnts = [0] * len(bin_sizes)

          head_labels = np.copy(labels)
          inp_pos_per_bin, tgt_pos_per_bin = [], []
          for b, (left, right) in enumerate(zip(cutoffs[1:-1], cutoffs[2:])):
            inp_pos = np.where((inputs >= left) * (inputs < right))[0]
            tgt_pos = np.where((labels >= left) * (labels < right))[0]
            inp_pos_per_bin.append(inp_pos)
            tgt_pos_per_bin.append(tgt_pos)

            head_labels[tgt_pos] = cutoffs[1] + b

          feature["head_labels"] = _int64_feature(head_labels)

          # permutation feature
          def _add_perm_feature(feature, pos_per_bin, cnts, prefix):
            for b, pos in enumerate(pos_per_bin):
              idx_tuple = []
              for p in pos:
                if cnts[b] < bin_sizes[b]:
                  idx_tuple.append([p, cnts[b]])
                  cnts[b] += 1
                else:
                  break

              n_tup = len(idx_tuple)
              tup = np.array(idx_tuple).reshape(n_tup * 2)

              feature["{}_cnt_{}".format(prefix, b)] = _int64_feature([n_tup])
              feature["{}_tup_{}".format(prefix, b)] = _int64_feature(tup)

          _add_perm_feature(feature, inp_pos_per_bin, inp_cnts, "inp")
          _add_perm_feature(feature, tgt_pos_per_bin, tgt_cnts, "tgt")

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        record_writer.write(example.SerializeToString())

      num_batch += 1

    record_writer.close()
    print("Done writing {}. batches: {}".format(file_name, num_batch))

    return file_name, num_batch


def batchify(data, batch_size, num_passes):
  """
    if use_tpu = True: num_passes > 1 
    
    Since TPU training requires entire [bsz x tgt_len] chunks, it can discard
    as many as `bsz * tgt_len` tokens in training. When `bsz` and `tgt_len` are 
    both large, as in the case of TPU training for Transformer-XL, the problem
    may lead to detectable performance drop. 

    Here, we use multiple randomly shifted copies to deal with this problem.
  """
  if num_passes > 1:
    data_len = len(data)
    double_data = np.concatenate([data, data])
    data_list = []
    for i in range(num_passes):
      start = np.random.randint(0, data_len)
      data_list.append(double_data[start:start+data_len])
    data = np.concatenate(data_list)

  num_step = len(data) // batch_size
  data = data[:batch_size * num_step]
  data = data.reshape(batch_size, num_step)

  return data


def get_bin_sizes(data, batch_size, tgt_len, cutoffs, std_mult=[2.5, 2.5, 2.5]):
  """
  Note: the `batch_size` here should be per-core batch size
  """
  bin_sizes = []

  def _nearest_to_eight(x): # so that it's faster on TPUs
    y = x - x % 8
    return y + 8 if x % 8 >= 4 else max(8, y)

  if cutoffs:
    num_batch = len(data) // batch_size // tgt_len

    data = data[:batch_size * num_batch * tgt_len]
    data = data.reshape(batch_size, num_batch, tgt_len)

    tot = batch_size * tgt_len
    for b, (left, right) in enumerate(zip(cutoffs[1:-1], cutoffs[2:])):
      mask = (data >= left) * (data < right)
      percents = mask.astype(np.float64).sum(2).sum(0) / tot
      mean = np.mean(percents)
      std = np.std(percents)

      bin_size = int(math.ceil(tgt_len * batch_size * (mean + std_mult[b] * std)))
      bin_size = _nearest_to_eight(bin_size)
      bin_sizes.append(bin_size)

  return bin_sizes

######################################################
######################################################


# Preprocess [features]
# =====================
# (data is in correct format but will be embelished)
#
#
#
#
#
#
#
#
#
#
#
#
#
#



# Train model
# ===========
# - Load corpus metadata to dict
# - Load record info
# - Extract from arguments batch size, data directory
# - In a input_fn:
#   - Load dataset from tensor slices (tf records) to TFRecordDataset
#   - parse dataset row by row
#   - batch data, shuffle and prefetch 
#       - Prefetch allows later elements to be prepared whilst current element is processed
# - In a model_fn:
#   - transpose features
#   - Initialise (presumably model weights) with uniform or normal
#   - Instantiate transformer model
#   - record mean loss
#   - configure step, learning rate, params,  optimiser and solver
#   
#
#
#
#
#


# Predict

# Output






    