



# Bumblebeat

Transformer NN architecture experimentation on the Magenta Groove Midi dataset for automated drum pattern generation. This code is an adaptation of the PyTorch implementation of the [Transformer-XL network](https://github.com/kimiyoung/transformer-xl) [2].

## 1. Dataset

This project relies on the [Groove Midi Dataset](https://magenta.tensorflow.org/datasets/groove#dataset), the code handles downloading and processing. For more information visit the embedded link.

## 2. Installation

To install Bumblebeat and all dependencies...

`pip install -e .`

Ensure that your pip version and python version are in sync.

### 2.1 Getting Started on AWS
The script `scripts/aws_init.sh` contains all you need to setup the development environment on AWS with the *Deep Learning Base AMI (Amazon Linux 2) Version 31.0*. Since some of the installs in the script prompt the user for Y/N, it is recommended to run the script in stages rather than as one.

Some useful one liners for interacting with the instance - such as ssh, file transfer, directory sync - can be found in `scripts/instance.sh`. 

## 3. Usage

### 3.1 Configuration Files

All pipeline configuration is specified in YAML files (typically found in the `conf/` folder) passed to the the CLI (**3.2 CLI commands**) .

There are three YAMLs required to run the pipeline...

`conf` - This is where all pipeline parameters are specified; data, modelling and evaluation. The default files are `conf/train_conf.yaml` and `conf/test_conf.yaml` , you should edit these rather than create your own. More information about each parameter is found within these files. It is important to only pass one to the pipeline at once.

`pitches` - Here lies a list of lists indicating how pitch classes are grouped. Classes in the sublist are considered the same instrument by the pipeline and are tokenised identically. This allows us to reduce complexity easily. The default for this is found in `conf/drum_pitches.yaml`

`time_steps` - This is a dict of mappings between number of ticks (the minimum unit of time in midi sequences) and time tokens. Time tokens are tokens that do not symbolise a pitch but instead, silence. This dict is used to tokenise silences in our midi sequences by replacing silences with the least amount of time tokens possible corresponding to the number of ticks. The default time_steps dict is found at `conf/time_steps_vocab.yaml`.

### 3.2 CLI commands

To test the data processing is functioning correctly you can run...

```
python bumblebeat data-pipeline \
    --conf-path 'conf/train_conf.yaml' \
    --pitches-path 'conf/drum_pitches.yaml' \
    --time-steps-path 'conf/time_steps_vocab.yaml'
```

And batched train data will be printed to screens. (Alter paths as necessary)


To train...

```
python bumblebeat model-pipeline \
    --conf-path 'conf/train_conf.yaml' \
    --pitches-path 'conf/drum_pitches.yaml' \
    --time-steps-path 'conf/time_steps_vocab.yaml'
```

To evaluate...

```
python bumblebeat model-pipeline \
    --conf-path 'conf/train_conf.yaml' \
    --pitches-path 'conf/drum_pitches.yaml' \
    --time-steps-path 'conf/time_steps_vocab.yaml'
```

(note that the difference in the last two steps should be whether or not `do_train` or `do_eval` are True or False in the primary conf)

To generate from a trained model see `bumblebeat/output/generate.py`. Specifically, `generate_sequences()`, `continue_sequence()` and `accompany_sequence()`.  CLI functionality coming soon...


## 4. References

[1] Jon Gillick, Adam Roberts, Jesse Engel, Douglas Eck, and David Bamman.
"Learning to Groove with Inverse Sequence Transformations."
  International Conference on Machine Learning (ICML), 2019.

[2] Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov - Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context

[3] Chris Donahue, Huanru Henry Mao, Yiting Ethan Li, Garrison W. Cottrell, Julian McAuley - LakhNES: Improving multi-instrumental music generation with cross-domain pre-training
