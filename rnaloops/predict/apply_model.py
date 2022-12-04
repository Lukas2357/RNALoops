"""Module to apply prepared sequential model on prepared RNALoops data """

from timeit import default_timer

import joblib
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from rnaloops.predict.prepare_model import HaltCallback


def fit_model(model, ax=None, print_model=False):
    """Fit a predefined sequential model and plot the training process

    Parameters
    ----------
    model : prepare_model.MyModel
        A custom model dataclass as created by prepare_model.prep_model
        Make sure to set all of its attributes properly before using this function
    ax : plt.Axes
        The axis to plot the training in
    print_model : bool, default False
        Whether to print out the model summary

    Returns
    -------
    prepare_model.MyModel
        The input model modified and extended with training parameters

    """
    # This is the callback to stop training if mse < 10^-4
    stop_training = HaltCallback(threshold=model.threshold)

    # Split train and test data. Note that this is a trivial split. It will not solve
    # the problem of intra- vs. inter-family cross-validation, as discussed in
    # https://doi.org/10.1093/bioinformatics/btac415
    # More effort is needed here...
    x_train, x_test, y_train, y_test = train_test_split(
        model.x_data, model.y_data, test_size=0.33, random_state=42
    )

    history = model.model.fit(
        x_train,
        y_train,
        epochs=model.epochs,
        verbose=0,
        validation_split=0.33,
        batch_size=model.batchsize,
        callbacks=stop_training,
    )

    if print_model:
        print(model.model.summary())

    loss_train = np.array(history.history["loss"])
    loss_valid = np.array(history.history["val_loss"])
    y_test_pred = [x[0] for x in model.model.predict(x_test, verbose=0)]

    plot_training(model, y_test, y_test_pred, loss_train, loss_valid, ax)

    model.mse_min = min(loss_valid)
    model.mse_med = np.median(loss_valid)
    model.stopped = len(loss_valid) < model.epochs

    return model


def plot_training(model, y_test, y_test_pred, loss_train, loss_valid, ax):
    """Specialised function to plot the training process

    Is called by fit_model and should not be used otherwise due to inflexibility.
    Modify the function here manually to adjust the plots.

    """
    y_test = list(y_test)
    n_plot = 100
    y_plot = range(len(y_test[:n_plot]))
    x_plot_1 = y_test[:n_plot]
    x_plot_2 = y_test_pred[:n_plot]
    mean_diff = np.median(abs(np.array(y_test_pred) - np.array(y_test)))

    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(16, 4), dpi=300)
        show_immediately = True
    else:
        show_immediately = False

    ax[0].plot(x_plot_1, y_plot, ".", label="data")
    ax[0].plot(x_plot_2, y_plot, ".", label="pred")
    for x1, x2, y in zip(x_plot_1, x_plot_2, y_plot):
        ax[0].plot([x1, x2], [y, y], "r-", label="_no_label_")
    ax[0].set_xlabel("Angle / °")
    ax[0].set_ylabel("Sequence")
    ax[0].legend()
    ax[0].set_title(f"Mean deviation in test data: {mean_diff:.2f} °")

    x = list(range(model.tot_epochs - model.epochs, model.tot_epochs))
    x_train = np.array(x)[: len(loss_train)]
    ax[1].plot(x_train, loss_train, "r-", label="mse train loss")
    x_valid = np.array(x)[: len(loss_valid)]
    ax[1].plot(x_valid, loss_valid, "g-", label="mse valid loss")

    ax[1].set_xlabel("epoches")
    ax[1].set_ylabel("mse")
    ax[1].set_ylim([0, 1.5 * max(loss_valid)])
    ax[1].legend()

    if show_immediately:
        plt.show()
        plt.close()


def converge_model(
    model,
    log_epochs=50,
    max_epochs=200,
    max_min=180,
    single_plot=True,
    print_model=False,
    save=True,
):
    """Converge the given model with the given parameters

    Parameters
    ----------
    model : prepare_model.MyModel
        A custom model dataclass as created by prepare_model.prep_model
    log_epochs : int, default 100
        The number of epochs after which training is logged and plotted
    max_epochs : int, default 1000
        Maximum number of epoches for training
    max_min : float, default 180
        maximum number of minutes for training
    single_plot : bool, default True
        Whether to show training plots as subplot in single plot. This will not allow
        to see the plots during training, but only after it
    print_model : bool, default False
        Whether to print out the model summary
    save : bool, default True
        Whether to save the result plot and model

    Returns
    -------
    prepare_model.MyModel
        The input model modified and extended by training

    """

    # To count epoches and time for possible stop of training:
    model.tot_epochs, elapsed_min, start, counter = 0, 0, default_timer(), 0
    # The number of epoches for training must be known to fit_model, so when we add
    # it to the model and pass this model to the fitting routine:
    model.epochs = log_epochs

    if single_plot:
        fig, ax = plt.subplots(
            rows := max_epochs // log_epochs, 2, figsize=(16, 4 * rows), dpi=300
        )
    else:
        ax, rows = None, 1
    while not model.stopped and model.tot_epochs < max_epochs and elapsed_min < max_min:
        model.tot_epochs += log_epochs
        axis = ax[counter] if single_plot else None
        # Force using CPU (forces use of all cores btw)
        with tf.device("/cpu:0"):
            model = fit_model(model, axis, print_model=print_model)
        counter += 1
        elapsed = (default_timer() - start) / 60
        print(
            f"{log_epochs / 1000:.0f}k epoches in {elapsed:.2f} min -> MSE min: "
            f"{model.mse_min:.1f} | MSE median: {model.mse_med:.1f}"
        )

    # If training was stopped before max_epoches, we remove other axis:
    if single_plot:
        for idx in range(rows - counter):
            ax[rows - idx - 1][0].remove()
            ax[rows - idx - 1][1].remove()

        plt.tight_layout()

        if save:
            plt.savefig("results/predict/model_training.png")

        plt.show()
        plt.close()

    print(f"Model with {model.model.count_params()} params fitted")

    if save:
        joblib.dump(model, "results/predict/my_model.pkl")

    return model