#!/usr/bin/env python
"""
This code is based on convolutional neural network model from below link
gist.github.com/jkleint/1d878d0401b28b281eb75016ed29f2ee
"""

from __future__ import print_function, division

import numpy as np
from keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten
from keras.models import Sequential
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import csv

import get_data

__date__ = '2018-05-16'

error_total = []
result = []
i = 0


def make_timeseries_regressor(window_size, filter_length, nb_input_series=1, nb_outputs=1, nb_filter=4):
    """:Return: a Keras Model for predicting the next value in a timeseries given a fixed-size lookback window of previous values.

    The model can handle multiple input timeseries (`nb_input_series`) and multiple prediction targets (`nb_outputs`).

    :param int window_size: The number of previous timeseries values to use as input features.  Also called lag or lookback.
    :param int nb_input_series: The number of input timeseries; 1 for a single timeseries.
      The `X` input to ``fit()`` should be an array of shape ``(n_instances, window_size, nb_input_series)``; each instance is
      a 2D array of shape ``(window_size, nb_input_series)``.  For example, for `window_size` = 3 and `nb_input_series` = 1 (a
      single timeseries), one instance could be ``[[0], [1], [2]]``. See ``make_timeseries_instances()``.
    :param int nb_outputs: The output dimension, often equal to the number of inputs.
      For each input instance (array with shape ``(window_size, nb_input_series)``), the output is a vector of size `nb_outputs`,
      usually the value(s) predicted to come after the last value in that input instance, i.e., the next value
      in the sequence. The `y` input to ``fit()`` should be an array of shape ``(n_instances, nb_outputs)``.
    :param int filter_length: the size (along the `window_size` dimension) of the sliding window that gets convolved with
      each position along each instance. The difference between 1D and 2D convolution is that a 1D filter's "height" is fixed
      to the number of input timeseries (its "width" being `filter_length`), and it can only slide along the window
      dimension.  This is useful as generally the input timeseries have no spatial/ordinal relationship, so it's not
      meaningful to look for patterns that are invariant with respect to subsets of the timeseries.
    :param int nb_filter: The number of different filters to learn (roughly, input patterns to recognize).
    """
    model = Sequential((
        # The first conv layer learns `nb_filter` filters (aka kernels), each of size ``(filter_length, nb_input_series)``.
        # Its output will have shape (None, window_size - filter_length + 1, nb_filter), i.e., for each position in
        # the input timeseries, the activation of each filter at that position.
        Convolution1D(nb_filter=nb_filter, filter_length=filter_length, activation='relu',
                      input_shape=(window_size, nb_input_series)),
        MaxPooling1D(),  # Downsample the output of convolution by 2X.
        Convolution1D(nb_filter=nb_filter * 2, filter_length=filter_length, activation='relu'),
        MaxPooling1D(),
        Flatten(),
        Dense(nb_outputs, activation='linear'),  # For binary classification, change the activation to 'sigmoid'
    ))
#     sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    # To perform (binary) classification instead:
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    return model


def make_timeseries_instances(timeseries, window_size):
    """Make input features and prediction targets from a `timeseries` for use in machine learning.

    :return: A tuple of `(X, y, q)`.  `X` are the inputs to a predictor, a 3D ndarray with shape
      ``(timeseries.shape[0] - window_size, window_size, timeseries.shape[1] or 1)``.  For each row of `X`, the
      corresponding row of `y` is the next value in the timeseries.  The `q` or query is the last instance, what you would use
      to predict a hypothetical next (unprovided) value in the `timeseries`.
    :param ndarray timeseries: Either a simple vector, or a matrix of shape ``(timestep, series_num)``, i.e., time is axis 0 (the
      row) and the series is axis 1 (the column).
    :param int window_size: The number of samples to use as input prediction features (also called the lag or lookback).
    """
    timeseries = np.asarray(timeseries)
    assert 0 < window_size < timeseries.shape[0]
    X = np.atleast_3d(
        np.array([timeseries[start:start + window_size] for start in range(0, timeseries.shape[0] - window_size)]))
    y = timeseries[window_size:]
    q = np.atleast_3d([timeseries[-window_size:]])
    return X, y, q


def evaluate_timeseries(timeseries, window_size, epoch=32):
    """Create a 1D CNN regressor to predict the next value in a `timeseries` using the preceding `window_size` elements
    as input features and evaluate its performance.

    :param ndarray timeseries: Timeseries data with time increasing down the rows (the leading dimension/axis).
    :param int window_size: The number of previous timeseries values to use to predict the next.
    :param int epoch: The epoch size of model.fit
    """
    filter_length = 5
    nb_filter = 4
    timeseries = np.atleast_2d(timeseries)
    if timeseries.shape[0] == 1:
        timeseries = timeseries.T  # Convert 1D vectors to 2D column vectors

    nb_samples, nb_series = timeseries.shape
    print('\n\nTimeseries ({} samples by {} series):\n'.format(nb_samples, nb_series), timeseries)
    model = make_timeseries_regressor(window_size=window_size, filter_length=filter_length, nb_input_series=nb_series,
                                      nb_outputs=nb_series, nb_filter=nb_filter)
    print('\n\nModel with input size {}, output size {}, {} conv filters of length {}'.format(model.input_shape,
                                                                                              model.output_shape,
                                                                                              nb_filter, filter_length))
    model.summary()

    error = []

    X, y, q = make_timeseries_instances(timeseries, window_size)
    print('\n\nInput features:', X, '\n\nOutput labels:', y, '\n\nQuery vector:', q, sep='\n')
    test_size = 30  # In real life you'd want to use 0.2 - 0.5
    X_train, X_test, y_train, y_test = X[:-test_size], X[-test_size:], y[:-test_size], y[-test_size:]
    history = model.fit(X_train, y_train, nb_epoch=epoch, batch_size=2, validation_data=(X_test, y_test))

    # serialize model to JSON
    model_json = model.to_json()
    global i
    with open("models\cnn_model" + str(i) + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("models\cnn_model" + str(i) + ".h5")
    print("Saved model to disk")
    i = i + 1

    pred = model.predict(X_test)
    print('\n\nactual', 'predicted', sep='\t')
    error_curr = 0
    for actual, predicted in zip(y_test, pred.squeeze()):
        print(actual.squeeze(), predicted, sep='\t')
        tmp = actual - predicted
        sum_squared = np.dot(tmp.T, tmp)
        error.append(np.sqrt(sum_squared))
        error_curr = error_curr + np.sqrt(sum_squared)
    print('next', model.predict(q).squeeze(), sep='\t')
    result.append(model.predict(q).squeeze())
    error_total.append(error_curr)
    print(error)

    return model, history

def main():
    """Prepare input data, build model, eval rate."""
    np.set_printoptions(threshold=25)
    ts_length = 1000
    window_size = 50
    number_of_runs = 5
    error_max = 200

    data = get_data.GetData()
    day_balance = data.get_day_balance()
    timeseries = np.array(day_balance['tBalance'])
    min_max_scalar = MinMaxScaler()
    timeseries = min_max_scalar.fit_transform(timeseries.reshape(-1,1))
    print(timeseries)

    model, history = evaluate_timeseries(timeseries, window_size, epoch=100)

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.show()

    # for i in range(number_of_runs):
    #     evaluate_timeseries(timeseries, window_size)

    # plt.plot(error_total)
    # plt.show()
    # print(result)
    #
    # error_total_new = []
    # for i in range(number_of_runs):
    #     if (error_total[i] < error_max):
    #         error_total_new.append(error_total[i])
    # plt.plot(error_total_new)
    # plt.show()
    # print(result)
    #
    # best_model = np.asarray(error_total).argmin(axis=0)
    # print("best_model=" + str(best_model))
    #
    # json_file = open('models\cnn_model' + str(best_model) + '.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = model_from_json(loaded_model_json)
    #
    # # load weights into new model
    # loaded_model.load_weights("models\cnn_model" + str(best_model) + ".h5")
    # print("Loaded model from disk")


if __name__ == '__main__':
    main()

    # print('\nSimple single timeseries vector prediction')
    # timeseries = np.arange(ts_length)  # The timeseries f(t) = t
    # enable below line to run this time series
    # evaluate_timeseries(timeseries, window_size)

    # print('\nMultiple-input, multiple-output prediction')
    # timeseries = np.array([np.arange(ts_length), -np.arange(ts_length)]).T  # The timeseries f(t) = [t, -t]
    # enable below line to run this time series
    ##evaluate_timeseries(timeseries, window_size)

    # print('\nMultiple-input, multiple-output prediction')
    # timeseries = np.array(
    #    [np.arange(ts_length), -np.arange(ts_length), 2000 - np.arange(ts_length)]).T  # The timeseries f(t) = [t, -t]
    # enable below line to run this time series
    # evaluate_timeseries(timeseries, window_size)
