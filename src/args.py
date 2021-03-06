#!/usr/bin/env python3
""" args.py
Udacity DSDN - Breno Silva
args.py command line argument definitions for train.py and test.py
"""

import argparse

def get_train_args(__author__, __version__, archs):
    """
    Get arguments for command line train.py
    """

    parser = argparse.ArgumentParser(
        description="Train and save an image classification model.",
        usage="python3 train.py flowers/ --learning_rate 0.003 --hidden_units 512 --epochs 10 --gpu",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'data_dir',
        action="store",
        help="Directory from where the training data is loaded"
    )

    parser.add_argument(
        '--save_dir',
        action="store",
        default=".",
        dest="save_dir",
        type=str,
        help="Directory to save checkpoints"
    )

    parser.add_argument(
        '--save_name',
        action="store",
        default="checkpoint",
        dest='save_name',
        type=str,
        help='Checkpoint filename',
    )

    parser.add_argument(
        '--categories_json',
        action="store",
        default="cat_to_name.json",
        dest='categories_json',
        type=str,
        help='Path to file containing the categories',
    )

    parser.add_argument(
        '--arch',
        action="store",
        default="densenet201",
        dest="arch",
        help="Supported architectures: " + ", ".join(archs)
    )

    parser.add_argument(
        '--gpu',
        action="store_true",
        default=False,
        dest="gpu",
        help="Enables the gpu mode"
    )

    hyperparameters = parser.add_argument_group('hyperparameters')

    hyperparameters.add_argument(
        '--learning_rate',
        action="store",
        default=0.003,
        dest="learning_rate",
        type=float,
        help="Value for learning rate"
    )

    hyperparameters.add_argument(
        '--hidden_units', '-hu',
        action="append",
        default=[],
        dest="hidden_size",
        type=int,
        help="Add repeated values to list"
    )

    hyperparameters.add_argument('--epochs',
        action="store",
        default=10,
        dest="epochs",
        type=int,
        help="Number of epochs"
    )

    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s ' + __version__ + ' [' + __author__ + ']'
    )

    return parser.parse_args()

def get_test_args(__author__, __version__):
    """
    Get arguments for command line test.py
    """

    parser = argparse.ArgumentParser(
        description="Predict an image with the classification model.",
        usage="python3 predict.py input checkpoint --category_names cat_to_name.json --gpu",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'input',
        action="store",
        help="Path to image that is going to be predicted"
    )

    parser.add_argument(
        'checkpoint',
        action="store",
        help="File containing the model checkpoint"
    )

    parser.add_argument(
        '--top_k',
        action="store",
        default=5,
        dest="top_k",
        type=int,
        help="Top K most likely classes"
    )

    parser.add_argument(
        '--categories_names',
        action="store",
        default="cat_to_name.json",
        dest='categories_names',
        type=str,
        help='Path to file containing the categories',
    )

    parser.add_argument(
        '--gpu',
        action="store_true",
        default=False,
        dest="gpu",
        help="Enables the gpu mode"
    )

    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s ' + __version__ + ' [' + __author__ + ']'
    )

    return parser.parse_args()