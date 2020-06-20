import functools
import json
import math
import os
import pickle

import numpy as np

from magenta import music as mm
from magenta.models.music_vae import data as vae_data

import tensorflow_datasets as tfds
from tensorflow.gfile import Exists as exists

import torch

import bumblebeat.utils.data as utils
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

    data_conf = conf['data']

    # ARGS in future
    dataset_name = data_conf['dataset']
    data_dir = data_conf['data_dir']

    train_batch_size = data_conf['per_host_train_bsz']

    corpus = get_corpus(
                dataset_name, 
                data_dir, 
                pitch_classes, 
                time_steps_vocab,
                conf['processing']
            )

    print ('-' * 10)
    print ('Train iterator')
    for batch in corpus.get_iterator('train', bsz=train_batch_size, bptt=100):
        print(batch)
        break

def get_corpus(dataset_name, data_dir, pitch_classes, time_steps_vocab, processing_conf):
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
    processing_conf: dict
        Dict of processing options

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
        bumblebeat.utils.data.create_dir_if_not_exists(fn)

        print("Producing dataset...")
        corpus = Corpus(
                    data_dir=data_dir,
                    dataset_name=dataset_name,
                    pitch_classes=pitch_classes, 
                    time_steps_vocab=time_steps_vocab,
                    processing_conf=processing_conf
                )
    
        print("Saving dataset...")
        with open(fn, "wb") as fp:
            pickle.dump(corpus, fp, protocol=2)

    return corpus



class LMShuffledIterator(object):
    def __init__(self, data, bsz, bptt, device='cpu', ext_len=None, shuffle=False):
        """
            data -- list[LongTensor] -- there is no order among the LongTensors
        """
        self.data = data

        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device
        self.shuffle = shuffle

    def get_sent_stream(self):
        # index iterator
        epoch_indices = np.random.permutation(len(self.data)) if self.shuffle \
            else np.array(range(len(self.data)))

        # sentence iterator
        for idx in epoch_indices:
            yield self.data[idx]

    def stream_iterator(self, sent_stream):
        # streams for each data in the batch
        streams = [None] * self.bsz

        data = torch.LongTensor(self.bptt, self.bsz)
        target = torch.LongTensor(self.bptt, self.bsz)

        n_retain = 0

        while True:
            # data   : [n_retain+bptt x bsz]
            # target : [bptt x bsz]
            data[n_retain:].fill_(-1)
            target.fill_(-1)

            valid_batch = True

            for i in range(self.bsz):
                n_filled = 0
                try:
                    while n_filled < self.bptt:
                        if streams[i] is None or len(streams[i]) <= 1:
                            streams[i] = next(sent_stream)
                        # number of new tokens to fill in
                        n_new = min(len(streams[i]) - 1, self.bptt - n_filled)
                        # first n_retain tokens are retained from last batch
                        data[n_retain+n_filled:n_retain+n_filled+n_new, i] = \
                            streams[i][:n_new]
                        target[n_filled:n_filled+n_new, i] = \
                            streams[i][1:n_new+1]
                        streams[i] = streams[i][n_new:]
                        n_filled += n_new
                except StopIteration:
                    valid_batch = False
                    break

            if not valid_batch:
                return

            data = data.to(self.device)
            target = target.to(self.device)

            yield data, target, self.bptt

            n_retain = min(data.size(0), self.ext_len)
            if n_retain > 0:
                data[:n_retain] = data[-n_retain:]
            data.resize_(n_retain + self.bptt, data.size(1))

    def __iter__(self):
        # sent_stream is an iterator
        sent_stream = self.get_sent_stream()
        for batch in self.stream_iterator(sent_stream):
            yield batch


class PartitionIterator(LMShuffledIterator):
    def __init__(self, raw_data, bsz, bptt, device='cpu', ext_len=None, shuffle=False):

        self.raw_data = iter(raw_data)

        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device
        self.shuffle = shuffle
    
    def get_batch(self, i, bptt=None):
        if bptt is None: bptt = self.bptt
        seq_len = min(bptt, self.data.size(0) - 1 - i)

        end_idx = i + seq_len
        beg_idx = max(0, i - self.ext_len)

        data = self.data[beg_idx:end_idx]
        target = self.data[i+1:i+1+seq_len]

        return data, target, seq_len

    def get_varlen_iter(self, start=0, std=5, min_len=5, max_deviation=3):
        max_len = self.bptt + max_deviation * std
        i = start
        while True:
            bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
            bptt = min(max_len, max(min_len, int(np.random.normal(bptt, std))))
            data, target, seq_len = self.get_batch(i, bptt)
            i += seq_len
            yield data, target, seq_len
            if i >= self.raw_data.size(0) - 2:
                break

    def __iter__(self):
        #if self.shuffle:
        #    np.random.shuffle(self.paths)
        # sents is list of tensors
        #if self.shuffle:
        #    np.random.shuffle(sents)
        for batch in self.stream_iterator(iter(self.raw_data)):
            yield batch

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
            processing_conf,
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
        self.processing_conf=processing_conf
        self.augment_stretch = True
        self.shuffle = True

        self.velocity_buckets = bumblebeat.utils.data.split_range(
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
        self.train = self.process_dataset(train_data, conf=processing_conf)
        print('Processing dataset TEST...')
        self.test = self.process_dataset(test_data, conf=processing_conf)
        print('Processing dataset VALID...')
        self.valid = self.process_dataset(valid_data, conf=processing_conf)

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

    def process_dataset(self, dataset, conf):
        """
        Augment, transform and tokenize each sample in <dataset>
        Return: list of tokenised sequences
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
    
    def get_iterator(self, split, *args, **kwargs):
        if split == 'train':
            data_iter = PartitionIterator(self.train, *args, **kwargs)
        elif split in ['valid', 'test']:
            data = self.valid if split == 'valid' else self.test
            data_iter = PartitionIterator(data, *args, **kwargs)

        return data_iter

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
              bumblebeat.utils.data.get_bucket_number(n.velocity, self.velocity_buckets),
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

        return torch.tensor(filled)

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
        time_vocab: d¡ict
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

    