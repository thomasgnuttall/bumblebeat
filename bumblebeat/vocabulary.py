import itertools

#def create_vocab(n_instruments, n_velocity_buckets, first_index=6):
#    """
#    Create vocabulary of all possible instrument-velocity combinations.

#    <first_index> dictates which index to start on, default 6 to allow for
#    0 to be special token indicating end and start of sequence and 1-5 to 
#    represent time steps in time_steps_vocab.yaml.

#    Each instrument is represented by <n_velocity_buckets> integers
#    Tokens increase across the dimension of instruments first:
#        token <first_index>     is instrument 0, velocity bucket 0
#        token <first_index> + 1 is instrument 0, velocity bucket 1
#        token <first_index> + 2 is instrument 0, velocity bucket 2
#        token <first_index> + N is instrument 0, velocity bucket 3
#            - where i = <n_instruments> - 1 (because of 0 indexing)
#            - where j = <n_velocity_buckets> - 1 (because of 0 indexing)
#            - where N = ixj - 1

#    returns: 2 x dict
#        {instrument_index: {velocity_index:token}}
#            ...for all instruments, velocities and tokens
#        {index: (instrument index, velocity index)},
#    """
#    
#    # itertools.product returns sorted how we desire    
#    all_comb = itertools.product(range(n_instruments), range(n_velocity_buckets))
#    d_reverse = {i+first_index:(x,y) for i,(x,y) in enumerate(all_comb)}
#    
#    d = {i:{} for i in range(n_instruments)}
#    for t, (i,v) in  d_reverse.items():
#        d[i][v] = t

#    return d, d_reverse


def create_vocab(n_instruments, first_index=16):
    """
    Create vocabulary of all possible instrument-velocity combinations.

    <first_index> dictates which index to start on, default 16 to allow for
    0 to be special token indicating end and start of sequence and 1-5 to 
    represent time steps in time_steps_vocab.yaml and 10 velocity buckets

    returns: 2 x dict
        {instrument_index: token}
            ...for all instruments and tokens
        {index: instrument_index}
    """
    d = {i:i+first_index for i in range(n_instruments)}
    d_reverse = {v:k for k,v in d.items()}
    return d, d_reverse