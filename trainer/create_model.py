import argparse
import inspect

from keras.preprocessing.text import Tokenizer
from model import create_model

tokenizer = Tokenizer(30000)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--train-file',
        help='GCS or local path to training data',
        required=True
    )
    parser.add_argument(
        '--num-epochs',
        help="""\
      Maximum number of training data epochs on which to train.
      If both --max-steps and --num-epochs are specified,
      the training job will run for --max-steps or --num-epochs,
      whichever occurs first. If unspecified will run for --max-steps.\
      """,
        type=int,
        default=200
    )
    parser.add_argument(
        '--train-batch-size',
        help='Batch size for training steps',
        type=int,
        default=40
    )
    parser.add_argument(
        '--dictionary-size',
        help='Size of dictionary used ',
        type=int,
        default=20000
    )
    parser.add_argument(
        '--eval-batch-size',
        help='Batch size for evaluation steps',
        type=int,
        default=40
    )

    parser.add_argument(
        '--eval-file',
        help='GCS or local path to evaluation data',
        required=True
    )
    # Training arguments
    parser.add_argument(
        '--embedding-size',
        help='Number of embedding dimensions for words',
        default=8,
        type=int
    )
    parser.add_argument(
        '--filters-count',
        help='Number of filters for CNN',
        default=100,
        type=int
    )

    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )

    args = parser.parse_args()
    arguments = args.__dict__

    print('Starting Census: Please lauch tensorboard to see results:\n'
          'tensorboard --logdir=$MODEL_DIR')

    # Run the training job
    # learn_runner pulls configuration information from environment
    # variables using tf.learn.RunConfig and uses this configuration
    # to conditionally execute Experiment, or param server code
    create_model(**arguments)
