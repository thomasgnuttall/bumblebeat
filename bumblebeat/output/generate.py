# Courtesy of chrisdonahue: https://github.com/kimiyoung/transformer-xl/issues/49
import torch
import torch.nn.functional as F
import numpy as np

from bumblebeat.transformer import MemTransformerLM

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
    Load pretrained Transformer model for
    auto-regressive prediction
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

