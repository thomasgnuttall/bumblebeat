import functools
import json
import math
import os
import pickle

import numpy as np

from magenta import music as mm
from magenta.models.music_vae import data as vae_data

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.gfile import Exists as exists

import bumblebeat.utils
import bumblebeat.vocabulary

def data_main(conf, pitch_classes, time_steps_vocab):
    """
    Run data pipeline
        Download data
        Process
        Store as TF Records

    Params
    ======
    conf: dict
        Dict of pipeline parameters
    pitch_classes: list
        list of lists indicating pitch class groupings
    time_steps_vocab: dict
        Dict of {number of ticks: token} for converting silence to tokens
    """

    # ARGS in future
    dataset_name = conf['dataset_name']
    data_dir = conf['data_dir']

    # ARGS for TF records
    per_host_test_bsz = conf['per_host_test_bsz']
    tgt_len = conf['tgt_len'] # number of steps to predict
    batch_size = conf['batch_size']

    corpus = get_corpus(dataset_name, data_dir)

    save_dir = os.path.join(data_dir, dataset_name, "tfrecords/")
    bumblebeat.utils.create_dir_if_not_exists(save_dir)

    # test mode
    # Here we want our data as a single sequence
    if per_host_test_bsz > 0:
        corpus.convert_to_tf_records("train", save_dir, tgt_len, batch_size)
        return

    for split, batch_size in zip(
      ["train", "valid"],
      [conf['per_host_train_bsz'], conf['per_host_valid_bsz']]):

        if batch_size <= 0: continue
        print("Converting {} set...".format(split))
        corpus.convert_to_tf_records(split, save_dir, tgt_len, batch_size)


def get_corpus(dataset_name, data_dir, pitch_classes, time_steps_vocab):
    """
    Load groove data into custom Corpus class
    
    Param
    =====
    dataset_name: str
        Name of groove dataset to download from tensorflow datasets
    data_dir: str
        Path to store data in (corpus, tf records)
    pitch_classes: list
        list of lists indicating pitch class groupings
    time_steps_vocab: dict
        Dict of {number of ticks: token} for converting silence to tokens

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
        bumblebeat.utils.create_dir_if_not_exists(fn)

        print("Producing dataset...")
        corpus = Corpus(
                    data_dir=data_dir,
                    dataset_name=dataset_name,
                    pitch_classes=pitch_classes, 
                    time_steps_vocab=time_steps_vocab
                )
    
        print("Saving dataset...")
        with open(fn, "wb") as fp:
            pickle.dump(corpus, fp, protocol=2)
        
        corpus_info = {
          "vocab_size" : corpus.vocab_size,
          "dataset" : corpus.dataset_name,
          'cutoffs': []
        }
        with open(os.path.join(data_dir, dataset_name, "corpus-info.json"), "w") as fp:
            json.dump(corpus_info, fp)

    return corpus


class Corpus:
    """
    Corpus to handle data in pipeline
    """
    def __init__(
            self, 
            data_dir, 
            dataset_name, 
            pitch_classes,  
            time_steps_vocab,
            n_velocity_buckets=10,
            min_velocity=0,
            max_velocity=127,
            augment_stretch=True,
            shuffle=True
        ):
        """
        Documentation here baby
        """
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.pitch_classes = pitch_classes
        self.time_steps_vocab = time_steps_vocab
        self.n_velocity_buckets = n_velocity_buckets
        self.augment_stretch = True
        self.shuffle = True

        self.velocity_buckets = bumblebeat.utils.split_range(
            min_velocity, max_velocity, n_velocity_buckets)
        self.pitch_class_map = self._classes_to_map(self.pitch_classes)
        self.n_instruments = len(set(self.pitch_class_map.values()))

        print(f'Generating vocab of {self.n_instruments} instruments and {n_velocity_buckets} velocity buckets')
        self.vocab, self.reverse_vocab = bumblebeat.vocabulary.create_vocab(
                        self.n_instruments, n_velocity_buckets, 
                        first_index=len(time_steps_vocab)
                    ) # leave initial indices for time steps vocab

        self.vocab_size = len(self.reverse_vocab) + len(time_steps_vocab) + 1 # add 1 for <eos> token

        train_data = self.download_midi(dataset_name, tfds.Split.TRAIN)
        test_data = self.download_midi(dataset_name, tfds.Split.TEST)
        valid_data = self.download_midi(dataset_name, tfds.Split.VALIDATION)
        #all_data = self.download_midi(dataset_name, tfds.Split.ALL)
        
        print('Processing dataset TRAIN...')
        self.train = self.process_dataset(train_data)
        print('Processing dataset TEST...')
        self.test = self.process_dataset(test_data)
        print('Processing dataset VALID...')
        self.valid = self.process_dataset(valid_data)
        #print('Processing dataset ALL...')
        #self.all = self.process_dataset(all_data)

        # TODO: Augment Data

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
        # Abstract to ARGS at some point
        quantize = conf['quantize']
        steps_per_quarter = conf['steps_per_quarter']
        filter_4_4 = conf['filter_4_4'] # maybe we dont want this?

        # To midi note sequence using magent
        dev_sequences = [mm.midi_to_note_sequence(features["midi"]) for features in dataset]
        
        if self.augment_stretch:
            augmented = self._augment_stretch(dev_sequences)
            # Tripling the total number of sequences
            dev_sequences = dev_sequences + augmented

        if quantize:
            dev_sequences = [self._quantize(d, steps_per_quarter) for d in dev_sequences]
        
        self.dev_sequences = dev_sequences

        # Filter out those that are not in 4/4 and do not have any notes
        dev_sequences = [
            s for s in dev_sequences 
            if self._is_4_4(s) and len(s.notes) > 0
            and s.notes[-1].quantized_end_step > mm.steps_per_bar_in_quantized_sequence(s)
        ]

        # note sequence -> [(pitch, vel_bucket, start timestep)]
        tokens = [self._tokenize(d, quantize=quantize) for d in dev_sequences]

        if self.shuffle:
            np.random.shuffle(tokens)

        return tokens
    
    def convert_to_tf_records(self, split, save_dir, tgt_len, batch_size):
        """
        Convert tensor data to TF records and store
        """
        # Our data is many small sequences,
        # we batch on a sample level
        data = getattr(self, split)

        file_names = []
        record_name = f"record_info-{split}.bsz-{batch_size}.tlen-{tgt_len}.json"
        record_info_path = os.path.join(save_dir, record_name)
        
        file_name, num_batch = create_ordered_tfrecords(save_dir, split, data, tgt_len, batch_size)

        file_names.append(file_name)

        with open(record_info_path, "w") as fp:
            record_info = {
              "filenames": file_names,
              "bin_sizes": [], # No bins here
              "num_batch": num_batch
            }
            json.dump(record_info, fp)

    def _augment_stretch(self, note_sequences):
        """
        Two stretchings for each sequence in <note_sequence>:
          - faster by 1%-10%
          - slower by 1%-10%
        These are returned as new sequences
        """
        augmented_slow = [mm.sequences_lib.augment_note_sequence(x, 1.01, 1.1, 0, 0)\
                        for x in note_sequences]
        augmented_fast = [mm.sequences_lib.augment_note_sequence(x, 0.9, 0.99, 0, 0)\
                        for x in note_sequences]
        return augmented_fast + augmented_slow

    def _quantize(self, s, steps_per_quarter=4):
        """
        Quantize a magenta Note Sequence object
        """
        return mm.sequences_lib.quantize_note_sequence(s, steps_per_quarter)

    def _is_4_4(self, s):
        """
        Return True if sample, <s> is in 4/4 timing, False otherwise
        """
        ts = s.time_signatures[0]
        return (ts.numerator == 4 and ts.denominator == 4)

    def _tokenize(self, note_sequence, quantize):
        """
        from magenta <note_sequence> return list of
        tokens, filling silence with time tokens in
        self.time_steps_vocab

        - if <quantized> use quantized_start_step else use start_time
        - pitch is mapped using self.pitch_class_map
        - velocities are bucketted as per self.velocity_buckets
        """
        d = [(self.pitch_class_map[n.pitch], \
              bumblebeat.utils.get_bucket_number(n.velocity, self.velocity_buckets),
              n.quantized_start_step if quantize else s.start_time) \
                for n in note_sequence.notes \
                if n.pitch in self.pitch_class_map]
        
        # Remove and fill list with silence vocab
        if quantize:
            #total_steps = note_sequence.total_quantized_steps
            #timestep_lim = self._roundup(total_steps, 4)
            #
            #filled = fill_timestep_silence(d, timestep_lim, self.time_steps_vocab)
        #else:
            ticks_per_quarter = note_sequence.ticks_per_quarter
            qpm = note_sequence.tempos[0].qpm # quarters per minute
            ticks_per_second = qpm*ticks_per_quarter/60

            filled = self._tokenize_w_ticks(d, ticks_per_second, self.vocab, self.time_steps_vocab)

        return filled

    def _tokenize_w_ticks(self, triples, ticks_per_second, pitch_vocab, time_steps_vocab):
        """
        From list of <triples> in the form:
            [(pitch class, bucketed velocity, start time (seconds)),...]
        Return list of tokens matching pitch-velocity combination to tokens in <pitch_vocab>
        and filling silence with time tokens in <time_steps_vocab>

        Returns: list
            sequence of tokens from the pitch and time_steps vocabularies
        """

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
            ticks = int(silence*ticks_per_second)
            if ticks:
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
                time_tokens = self._convert_num_to_denominations(ticks, time_steps_vocab)

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

        return w_silence

    def _convert_num_to_denominations(self, num, time_vocab):
        """
        Convert <num> into sequence of time tokens in (<time_vocab>).
        Tokens are selected so as to return a sequence of minimum length possible

        Params
        ======
        num: int
            Number of ticks to convert
        time_vocab: dict
            {num_ticks: token}

        Return
        ======
        list:
            [tokens representing number]
        """
        # Start with largest demoninations
        denominations = list(sorted(time_vocab.keys(), reverse=True)) 
        seq = []
        for d in denominations:
            div = num/d
            # If <num> can be divided by this number
            # Create as many tokens as possible with this denomination
            if div > 1:
                floor = math.floor(div)
                seq += floor*[time_vocab[d]]
                num -= floor*d
        return seq

    def _roundup(self, x, n):
        """
        Roundup <x> to nearest multiple of <n>
        """
        return int(math.ceil(x/n)) * n

    def _classes_to_map(self, classes):
        class_map = {}
        for cls, pitches in enumerate(classes):
            for pitch in pitches:
                class_map[pitch] = cls
        return class_map


def _int64_feature(values):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def create_ordered_tfrecords(save_dir, basename, data, tgt_len, batch_size):
    
    file_name = f"{basename}.bsz-{batch_size}.tlen-{tgt_len}.tfrecords"

    save_path = os.path.join(save_dir, file_name)
    record_writer = tf.python_io.TFRecordWriter(save_path)

    batched_data = batchify(data, batch_size)

    num_batch = 0
    # for t in range(0, batched_data.shape[1] - tgt_len - 1, tgt_len):
    for t in range(0, batched_data.shape[1] - 1, tgt_len):
      cur_tgt_len = min(batched_data.shape[1] - 1 - t, tgt_len)
      # drop the remainder if use tpu
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

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        record_writer.write(example.SerializeToString())

      num_batch += 1

    record_writer.close()
    print("Done writing {}. batches: {}".format(file_name, num_batch))

    return file_name, num_batch


def batchify(data, batch_size):
    """
    Create training batches
    """
    # Create one long sequence of data with individual samples
    # divided by end of sequence token, -1
    seq = functools.reduce(lambda x,y: x+[-1]+y, data)
    eos = max(seq) + 1 #  add new token for end of sequence
    seq = np.array([x if x != -1 else eos for x in seq])

    num_step = len(seq) // batch_size
    seq = seq[:batch_size * num_step]
    seq = seq.reshape(batch_size, num_step)

    return seq
    