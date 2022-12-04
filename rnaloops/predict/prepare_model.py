"""Module to prepare sequential NN for RNALoops"""

from dataclasses import dataclass
from typing import Iterable

import keras
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Dropout

from rnaloops.predict.prepare_input import prepare_data


class HaltCallback(tf.keras.callbacks.Callback):
    """Callback class to stop training when mse < threshold

    Use in model.fit as parameter like callback=HaltCallback() to stop the training
    when loss is below given threshold.

    """

    def __init__(self, threshold=10**-4):
        super().__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        """Stops training when loss is below given threshold at epoche end"""
        # Choose slightly lower value to be on the save side:
        if logs.get("loss") < 0.99 * self.threshold:
            print(f"\n##### MSE < {self.threshold} -> stop training! #####\n")
            self.model.stop_training = True


def make_model(neurons_per_layer, activation_fct, dropouts):
    """Create a simple model with dense and dropout layers

    Parameters
    ----------
    neurons_per_layer : Iterable
        Each entry represents number of neurons in that layer, thereby also sets
        number of layers
    activation_fct : str, any activation function known to keras
        The activation function applied to each layer
    dropouts : Iterable, same length as neurons_per_layer
        After each layer a dropout layer is added, where dropouts set the ratio of
        neurons to dropout. Set 0 to not use dropout in that layer.

    Returns
    -------
    keras.Sequential
        The model handle of keras sequential model

    """

    # If dropouts are missing we assume they should be 0:
    if (diff := len(neurons_per_layer) - len(dropouts)) > 0:
        dropouts = list(dropouts) + [0]*diff

    # These are the input, hidden and dropout layers:
    layers = []
    for npl, dropout in zip(neurons_per_layer, dropouts):
        layers.append(Dense(npl, activation=activation_fct))
        layers.append(Dropout(dropout))

    # For the output add simple linear neuron (regression task):
    model = Sequential(layers + [Dense(1)])

    # Adam performs best, steps_per_execution provide small performance boost
    model.compile(optimizer="Adam", loss=tf.keras.losses.MSE, steps_per_execution=16)

    return model


def prep_model(df, feature="planar_1", metric="mean"):
    """Prepare a model to be fitted immediately afterwards

    The model target will be the combination of the given feature and metric.
    All other model parameters are fixed in MyModel below. This is because prep_model
    is used for direct fitting of preselected model and not for model selection.

    Parameters
    ----------
    df : pd.DataFrame
        The aggregated RNALoops data. Its index must be either parts_seq or whole_seq
    feature : str
        The feature to use as target. So far only planar_X is supported with X being
        any integer from 1 to the loop_type of the aggregated df
    metric : str
        The aggregated df has mean and median values of angles as columns.
        Choose here which of the two to use as target.

    Returns
    -------
    MyModel
        Contains everything required to reconstruct and fit this model

    """
    # Will pad sequences and make dummies from each base:
    new_df = prepare_data(df, metric=metric)

    # The new_df has only planar angles as possible targets, if feature is not any of
    # planar_X this selection will fail. Modify prepare_data to accept more targets:
    y_data = new_df[f"{feature}_{metric}"]

    # make_dummies will create six columns for each base. They are labeled as
    # b<idx>_<base_entry>, where idx is the index of the base in the padded sequence
    # starting from the lowest base position in pdb chain order. Check prepare_sequence
    # for details on the padding:
    x_data = new_df[[c for c in new_df.columns if c[0] == "b" and len(c) < 7]]

    # Initialize a custom model and use make_model to build the tf model for it:
    model = MyModel(x_data, y_data)
    model.model = make_model(model.neurons_per_layer, "elu", model.dropouts)

    return model


@dataclass
class MyModel:
    """Isolate everything required to reconstruct and fit the model in this dataclass

    Change any attribute here to change the default model behaviour

    """

    x_data: pd.DataFrame  # The features
    y_data: pd.Series  # The target
    neurons_per_layer: Iterable = tuple(range(100, 2, -5))  # Neurons per layer
    dropouts: Iterable = (0.2, )  # The dropout ratios (index=layer)
    activation: str = "elu"  # Can be any activation function known by keras
    mse_min: float = 10**6  # The minimum mse found in training so far
    mse_median: float = 10**6  # The median mse found in last period
    stopped: bool = False  # True if the model was stopped in last period
    batchsize: int = 64  # The batch size for model training
    threshold: int = 10**-4  # The threshold mse below which training is stopped
    model: keras.Sequential = None  # make_model will return such a model
    tot_epochs: int = 0  # The number of epochs fitted so far