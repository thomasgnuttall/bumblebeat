    import click
    
    #from bumblebeat.data import data_main
    from bumblebeat.utils.data import load_yaml
    from bumblebeat.data import get_corpus
    
    import random
    from bumblebeat.output.generate import *

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
        time_vocab,
        conf['processing']
    )
    pitch_vocab = corpus.reverse_vocab
    velocity_vocab = {v:k for k,v in corpus.vel_vocab.items()}
    device = 'cpu'
    path = 'train_step_32000/model.pt'
    model = load_model(path, device)

    USE_CUDA = False
    mem_len = 1000
    gen_len = 500
    same_len = True
    
    simplified_pitches = [[36], [38], [42], [46], [45], [48], [50], [49], [51]]
    device = torch.device("cuda" if USE_CUDA else "cpu")

    random_sequence = random.choice([x for x in corpus.train_data if x['style']['primary']==7])
    
    for i in [4, 8, 16]:
        if i:
            # To midi note sequence using magent
            dev_sequence = corpus._quantize(mm.midi_to_note_sequence(random_sequence["midi"]), i)
            quantize=True
        else:
            dev_sequence = mm.midi_to_note_sequence(random_sequence["midi"])
            quantize=False
        
        # note sequence -> [(pitch, vel_bucket, start timestep)]
        in_tokens = corpus._tokenize(dev_sequence, i, quantize)
        note_sequence = tokens_to_note_sequence(
                in_tokens, 
                pitch_vocab, 
                simplified_pitches, 
                velocity_vocab, 
                time_vocab,
                random_sequence['bpm'])
        note_sequence_to_midi_file(note_sequence, f'sound_examples/experiments/original_quantize={i}.midi')

    out_tokens = continue_sequence(
                    model,
                    seq=in_tokens[-1000:],
                    prime_len=512,
                    gen_len=gen_len, 
                    mem_len=mem_len, 
                    device=device,
                    temp=0.95, 
                    topk=32)

    note_sequence = tokens_to_note_sequence(
            out_tokens, 
            pitch_vocab, 
            simplified_pitches, 
            4, 
            time_vocab, 
            random_sequence['bpm'])
    note_sequence_to_midi_file(note_sequence, f'sound_examples/experiments/continued.midi')


def count_ticks(seq, reverse_time_vocab):
    return sum([reverse_time_vocab[s] for s in seq if s in reverse_time_vocab.keys()])
































triples = [(corpus.pitch_class_map[n.pitch], \
      bumblebeat.utils.data.get_bucket_number(n.velocity, corpus.velocity_buckets), \
      n.quantized_start_step if quantize else n.start_time) \
        for n in note_sequence.notes \
        if n.pitch in corpus.pitch_class_map]

ticks_per_quarter = note_sequence.ticks_per_quarter
qpm = note_sequence.tempos[0].qpm # quarters per minute
ticks_per_second = qpm*ticks_per_quarter/60




w_silence = []

# Initalise counter to keep track of consecutive pitches
# so that we can ensure they are appended to our
# final tokenised sequence in numerical order
consecutive_pitches = 0

# index, (pitch, velocity, start time)
for i, (x, y, z) in enumerate(triples[:5]):
    if i == 0:
        silence = z
    else:
        silence = z - triples[i-1][2] # z of previous element

    if quantize:
        ticks = silence*ticks_per_quarter/steps_per_quarter
    else:
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

    import ipdb; ipdb.set_trace()
    # Triple to tokens...
    #   Discard time since we have handled that with time tokens.
    #   Look up pitch velocity combination for corresponding token.
    pitch_tok = corpus.vocab[x][y] # [pitch class][velocity]
    w_silence.append(pitch_tok)



























































import pretty_midi

# Load MIDI file into PrettyMIDI object
midi_data = pretty_midi.PrettyMIDI('sound_examples/experiments/basic_dancehall.mid')
dev_sequence = corpus._quantize(mm.midi_to_note_sequence(midi_data), 4)
seq = corpus._tokenize(dev_sequence, 4, True)

reverse_time_vocab = {v:k for k,v in time_vocab.items()}

gen_len = len(seq) + 1
seq = [0] + seq
prime_len = 380

assert gen_len <= len(seq), "Cannot accompany beyond length of input sequence"
sampler = TxlSimpleSampler(model, device, mem_len=mem_len)



#inp, sampler = prime_sampler(sampler, seq, prime_len)
inp = 0
nll = 0.
rhythm = []
for i in range(gen_len):
    _, probs = sampler.sample_next_token_updating_mem(seq[i], exclude_eos=False)
    _probs = probs.cpu().numpy()
    
    #_probmask = np.zeros_like(_probs)
    #_probmask[seq] = 0
    #_probs *= _probmask
    
    if topk is not None:
        ind = np.argpartition(_probs, -topk)[-topk:]
        _probmask = np.zeros_like(_probs)
        _probmask[ind] = 1.
        _probs *= _probmask
    
    _probs /= np.sum(_probs)
    tar = np.random.choice(corpus.vocab_size, p=_probs)

    if seq[i] != 0:
        rhythm.append(seq[i])
        if tar not in list(time_vocab.values())+[0]:
            rhythm.append(tar)
            _, probs = sampler.sample_next_token_updating_mem(tar, exclude_eos=False)
    inp = tar



bpm = midi_data.get_tempo_changes()[-1][0]
time_sig_denominator = midi_data.time_signature_changes[0].denominator
qpm = bpm/(time_sig_denominator/4)
ticks_per_second = midi_data.time_to_tick(1)

ticks_per_quarter = int(ticks_per_second*60/qpm)

out_tokens = rhythm
note_sequence = tokens_to_note_sequence(
        out_tokens, 
        pitch_vocab, 
        simplified_pitches, 
        4, 
        time_vocab, 
        qpm,
        ticks_per_quarter=ticks_per_quarter)
note_sequence_to_midi_file(note_sequence, f'sound_examples/experiments/augmented.midi')























import pretty_midi

# Load MIDI file into PrettyMIDI object
midi_data = pretty_midi.PrettyMIDI('sound_examples/experiments/basic_dancehall.mid')
dev_sequence = corpus._quantize(mm.midi_to_note_sequence(midi_data), 4)
seq = corpus._tokenize(dev_sequence, 4, True)

reverse_time_vocab = {v:k for k,v in time_vocab.items()}

gen_len = len(seq) + 1
seq = [0] + seq

assert gen_len <= len(seq), "Cannot accompany beyond length of input sequence"
sampler = TxlSimpleSampler(model, device, mem_len=mem_len)


inp = 0
nll = 0.
rhythm = []
for i in range(gen_len):
    if seq[i] in silence_tokens:
        sampler.sample_next_token_updating_mem(inp, exclude_eos=False)
    else:
        _, probs = sampler.sample_next_token_updating_mem(inp, exclude_eos=False)
        _probs = probs.cpu().numpy()
        
        #_probmask = np.zeros_like(_probs)
        #_probmask[seq] = 1.
        #_probs *= _probmask
        
        if topk is not None:
            ind = np.argpartition(_probs, -topk)[-topk:]
            _probmask = np.zeros_like(_probs)
            _probmask[seq] = 1.
            _probs *= _probmask
        
        _probs /= np.sum(_probs)
        tar = np.random.choice(corpus.vocab_size, p=_probs)
        #assert tar in seq
    rhythm.append(tar)

    inp = tar

bpm = midi_data.get_tempo_changes()[-1][0]
time_sig_denominator = midi_data.time_signature_changes[0].denominator
qpm = bpm/(time_sig_denominator/4)
ticks_per_second = midi_data.time_to_tick(1)

ticks_per_quarter = int(ticks_per_second*60/qpm)

out_tokens = rhythm
note_sequence = tokens_to_note_sequence(
        seq[1:], 
        pitch_vocab, 
        simplified_pitches, 
        4, 
        time_vocab, 
        qpm,
        ticks_per_quarter=ticks_per_quarter)
note_sequence_to_midi_file(note_sequence, f'sound_examples/experiments/augmented.midi')






















out_tokens = accompany_sequence(model, tokens, list(time_steps_vocab.keys()), gen_len=len(tokens), temp=0.95, topk=32, mem_len=mem_len, device=device)
note_sequence = tokens_to_note_sequence(
        out_tokens, 
        pitch_vocab, 
        simplified_pitches, 
        10, 
        time_vocab, 
        midi_data['bpm'])
note_sequence_to_midi_file(note_sequence, f'sound_examples/experiments/augmented.midi')
