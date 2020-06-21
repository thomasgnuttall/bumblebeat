import torch
import torch.nn.functional as F
import numpy as np

from mem_transformer import MemTransformerLM

MODEL_FP = 'pretrained/model.pt'
USE_CUDA = True
BATCH_SIZE = 1
TGT_LEN = 1
EXT_LEN = 0
MEM_LEN = 2000
CLAMP_LEN = 1000
GEN_LEN = 4000
SAME_LENGTH = True

device = torch.device("cuda" if USE_CUDA else "cpu")

# Load the best saved model
with open(MODEL_FP, 'rb') as f:
    model = torch.load(f)
model.backward_compatible()
model = model.to(device)

# Make sure model uses vanilla softmax
if model.sample_softmax > 0:
  raise NotImplementedError()
if model.crit.n_clusters != 0:
  raise NotImplementedError()

# Change training length/memory attrs
model.reset_length(TGT_LEN, EXT_LEN, MEM_LEN)
if CLAMP_LEN > 0:
  model.clamp_len = CLAMP_LEN
if SAME_LENGTH:
  model.same_length = True

# Turn on evaluation mode which disables dropout.
model.eval()

# Generate sequences of specified length and number
with torch.no_grad():
  # Create buffer for generated sequences
  samples = torch.zeros([0, BATCH_SIZE], dtype=torch.int64).to(device)

  # Initialize state
  prev_token = torch.zeros([TGT_LEN, BATCH_SIZE], dtype=torch.int64).to(device)
  mems = tuple()

  # Autoregressive sampling
  for i in range(GEN_LEN):
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

# Should be [GEN_LEN, BATCH_SIZE]
print(samples.shape)