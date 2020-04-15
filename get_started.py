# coding: utf-8
import argparse
import time
import math
import os, sys
import itertools

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from model.data_utils import get_lm_corpus
from model.mem_transformer import MemTransformerLM
from model.utils.exp_utils import create_exp_dir
from model.utils.data_parallel import BalancedDataParallel

device = torch.device('cpu')


# Load data (return class)

# Preprocess and Create TF Records
# ================================
# (Get raw data into format ready for transformer)
# 
# - Different depending on CPU, GPU or TPU
# - Wrap in pipeline ready Corpus class
#	- Identify vocabulary (class of its own) 
#		- Count and identify unique "tokens"
#		- ¿Vocabs consistent across datasets?
#		- Important to consider encoding of "special" tokens
#		- Attribute of the corpus
#	- Store vocab as file associated with dataset (for future)
#	- Load vocab
#	- Create dictionary of index to symbol and vice versa (as attribute of vocab)
#	- Consider filtering out uncommon tokens (set max_size of vocab)
# 	- Load and encode data into numpy array (replace tokens with index)
# 		- For train, test and validation (each an attribute of the corpus)
# 	- Pickle corpus
# 	- Pickle alongside metadata of corpus in dictionary
# - Seperately store train,test, validation to tensorflow records
# 	- With accompanying metadata json
#	- Binary storage format of tensorflow
# 		- useful for streaming data over a network
#		- useful for aggregating datasets
#		- integrates with TF nicely
#		- datasets arent stored in RAM
#		- Very efficient data import for sequence data


class Corpus:
	"""
	Corpus to handle data in pipeline
	"""
	def __init__(self, dataset):
		self.dataset = dataset
		self.train = None
		self.test = None
		self.val = None

		pass

	def train_test_split(self):
		pass

	def get_sample(self, prop):
		pass

    def __getattr__(self, attr):
        return attr

def load_data_from_path(path):
	dataset = None
	return Corpus(dataset=dataset)

path = 'path_to_data/data.something'
corpus = load_data_from_path(path=path)

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

import tensorflow as tf
import tensorflow_datasets as tfds


dataset = tfds.load(
    name=path, 
    split=tfds.Split.TRAIN,
    try_gcs=True)

# Build your input pipeline
dataset = dataset.shuffle(1024).batch(32).prefetch(
    tf.data.experimental.AUTOTUNE)

for features in dataset.take(1):
    # Access the features you are interested in
    midi, genre = features["midi"], features["style"]["primary"]


# Train model
# ===========
# - Load corpus metadata to dict
# - Load record info
# - Extract from arguments batch size, data directory
# - In a input_fn:
# 	- Load dataset from tensor slices (tf records) to TFRecordDataset
# 	- parse dataset row by row
# 	- batch data, shuffle and prefetch 
#		- Prefetch allows later elements to be prepared whilst current element is processed
# - In a model_fn:
#	- transpose features
#	- Initialise (presumably model weights) with uniform or normal
#	- Instantiate transformer model
#	- record mean loss
#	- configure step, learning rate, params,  optimiser and solver
#	
#
#
#
#
#


# Predict

# Output






