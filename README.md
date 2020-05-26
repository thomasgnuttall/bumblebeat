


# Bumblebeat

Transformer NN architecture experimentation on the Magenta Groove Midi dataset for automated drum pattern generation. This code is an adaptation of the tensorflow implementation of the [Transformer-XL network](https://github.com/kimiyoung/transformer-xl) [2].

## 1. Dataset

This project relies on the [Groove Midi Dataset](https://magenta.tensorflow.org/datasets/groove#dataset), the code handles downloading and processing but for more information visit the embedded link.

## 2. Installation

To install Bumblebeat and all dependencies...

`pip install -e .`

Ensure that your pip version and python version are in sync.

### 3.1 Getting Started on AWS
The script `scripts/aws_init.sh` contains all you need to setup the development environment on AWS with the *Deep Learning Base AMI (Amazon Linux 2) Version 24.0*. It is recommended to run the script in stages rather than as one.

It is important to note that if you intend to rely on GPUs you must make sure the correct requirement is uncommented in `requirements.txt` (more details there).

## 3. Usage

### 3.1 Configuration Files

All pipeline configuration is specified in YAML files (typically found in the `conf/` folder) passed to the the CLI (**3.2 CLI commands**) .

There are three YAMLs required to run the pipeline...

`conf` - This is where all pipeline parameters are specified; data, modelling and evaluation. The default files are `conf/train_conf.yaml` and `conf/test_conf.yaml` , you should edit these rather than create your own. More information about each parameter is found within these files. It is important to only pass one to the pipeline at once.

`pitches` - Here lies a list of lists indicating how pitch classes are grouped. Classes in the sublist are considered the same instrument by the pipeline and are tokenised identically. This allows us to reduce complexity easily. The default for this is found in `conf/drum_pitches.yaml`

`time_steps` - This is a dict of mappings between number of ticks (the minimum unit of time in midi sequences) and time tokens. Time tokens are tokens that do not symbolise a pitch but instead, silence. This dict is used to tokenise silences in our midi sequences by replacing silences with the least amount of time tokens possible corresponding to the number of ticks. The default time_steps dict is found at `conf/time_steps_vocab.yaml`.

### 3.2 CLI commands

To download, process and store data. Specify desired parameters in `data` section of  `conf` and run...

```
python bumblebeat data-pipeline \
    --conf-path 'conf/train_conf.yaml' \
    --pitches-path 'conf/drum_pitches.yaml' \
    --time-steps-path 'conf/time_steps_vocab.yaml'
```

(alter paths as necessary)


To train...

```
python bumblebeat model-pipeline --conf-path 'conf/train_conf.yaml'
```

To evaluate...

```
python bumblebeat model-pipeline --conf-path 'conf/test_conf.yaml'
```
(note that the difference in the last two steps is whether or not `do_train` or `do_eval` are True or False)

To generate...

`bash here`

To evaluate...

`bash here`


## 4. References

[1] Jon Gillick, Adam Roberts, Jesse Engel, Douglas Eck, and David Bamman.
"Learning to Groove with Inverse Sequence Transformations."
  International Conference on Machine Learning (ICML), 2019.

[2] Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov - Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context