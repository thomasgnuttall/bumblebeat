import click

from bumblebeat.data import data_main
from bumblebeat.utils import load_yaml

@click.group()
def cli():
    pass


@cli.command(name="data-pipeline")
@click.argument('conf_path', type=str, default='conf/train_conf.yaml', required=False)
@click.argument('pitches_path', type=str, default='conf/drum_pitches.yaml', required=False)
@click.argument('time_steps_path', type=str, default='conf/time_steps_vocab.yaml', required=False)
def cmd_run_pipeline(conf_path, pitches_path, time_steps_path):
    """
    Run data pipeline,
        Download data
        Process
        Store as TF Records
    """
    conf = load_yaml(conf_path)
    
    pitch_classes_yaml = load_yaml(pitches_path)
    pitch_classes = pitch_classes_yaml['DEFAULT_DRUM_TYPE_PITCHES']

    time_steps_vocab = load_yaml(time_steps_path)

    data_main(conf, pitch_classes, time_steps_vocab)


if __name__ == '__main__':
    cli()