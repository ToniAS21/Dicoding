"""
Author: Toni Andreas Susanto
Date: 23/01/2023
This is the transform.py module.
Usage:
- Function Tranform Data
"""

import tensorflow as tf
import tensorflow_transform as tft

CATEGORICAL_FEATURES = {
    'eligible': 2,
    'job': 12,
    'marital': 3,
    'education': 4,
    'targeted': 2,
    'default': 2,
    'housing': 2,
    'loan': 2,
    'month': 12
}

NUMERICAL_FEATURES = [
    'age',
    'salary',
    'balance',
    'duration',
    'day',
    'campaign',
    'pdays',
    'previous'
]

LABEL_KEY = 'response'


def transformed_name(key):
    """Transform feature key

    Args:
        key (str): the key to be transformed

    Returns:
        str: transformed key
    """

    return f"{key}_xf"


def convert_num_to_one_hot(label_tensor, num_labels=2):
    """Convert a label (0 or 1) into a one-hot vector

    Args:
        label_tensor (int): label tensor (0 or 1)
        num_labels (int, optional): num of label. Defaults to 2.

    Returns:
        tf.Tensor: label tensor
    """

    one_hot_tensor = tf.one_hot(label_tensor, num_labels)
    return tf.reshape(one_hot_tensor, [-1, num_labels])


def preprocessing_fn(inputs):
    """Preprocess input features into transformed features

    Args:
        inputs (dict): map from feature keys to raw features

    Returns:
        dict: map from features keys to transformed features
    """

    outputs = {}

    for keys, values in CATEGORICAL_FEATURES.items():
        int_value = tft.compute_and_apply_vocabulary(
            inputs[keys], top_k=values + 1)
        outputs[transformed_name(keys)] = convert_num_to_one_hot(
            int_value, num_labels=values + 1)

    for feature in NUMERICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.scale_to_0_1(inputs[feature])

    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
