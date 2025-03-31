#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
import tflearn
import logging


class NeuralNetWork:
    def __init__(self, feature_number, rows, columns, layers, device):
        tf_config = tf.ConfigProto()
        self.session = tf.Session(config=tf_config)
        if device == "cpu":
            tf_config.gpu_options.per_process_gpu_memory_fraction = 0
        else:
            tf_config.gpu_options.per_process_gpu_memory_fraction = 0.2
        self.input_num = tf.placeholder(tf.int32, shape=[])
        self.input_tensor = tf.placeholder(tf.float32, shape=[None, feature_number, rows, columns])
        self.previous_w = tf.placeholder(tf.float32, shape=[None, rows])
        self._rows = rows
        self._columns = columns

        self.layers_dict = {}
        self.layer_count = 0

        self.output = self._build_network(layers)

    def _build_network(self, layers):
        pass


class CNN(NeuralNetWork):
    # input_shape (features, rows, columns)
    def __init__(self, feature_number, rows, columns, layers, device):
        # Add feature_number to initialisation logging
        logging.debug(f"Initializing CNN with {feature_number} features")
        NeuralNetWork.__init__(self, feature_number, rows, columns, layers, device)

    def add_layer_to_dict(self, layer_type, tensor, weights=True):

        self.layers_dict[layer_type + '_' + str(self.layer_count) + '_activation'] = tensor
        self.layer_count += 1

    # generate the operation, the forward computaion
    def _build_network(self, layers):
        # Input: [batch, 29, 21, 6]
        network = self.input_tensor
        
        # Input validation
        tf.debugging.assert_rank(network, 4, message="Input must be 4D: [batch, assets, window, features]")
        tf.debugging.assert_equal(tf.shape(network)[2], 21, message="Window size must be 21")
        tf.debugging.assert_equal(tf.shape(network)[3], 6, message="Must have 6 input features")

        # Conv1D Processing
        network = tf.reshape(network, [-1, 21, 6])  # [batch*29, 21, 6]
        
        # Validate Conv1D input reshape
        original_elements = tf.shape(self.input_tensor)[0] * 29 * 21 * 6
        reshaped_elements = tf.shape(network)[0] * 21 * 6 
        tf.debugging.assert_equal(reshaped_elements, original_elements, 
                                message="Conv1D input reshape corrupted elements")

        network = tflearn.layers.conv_1d(
            network,
            nb_filter=64,
            filter_size=5,
            activation='relu',
            padding='valid'
        )  # [batch*29, 17, 64]

        # Validate Conv1D output
        expected_conv_elements = tf.shape(network)[0] * 17 * 64
        actual_conv_elements = tf.reduce_prod(tf.shape(network))
        tf.debugging.assert_equal(actual_conv_elements, expected_conv_elements,
                                message="Conv1D output shape mismatch")

        # LSTM Processing
        network = tflearn.layers.lstm(
            network,
            n_units=128,
            dropout=0.2,
            return_seq=False
        )  # [batch*29, 128]

        # Portfolio Allocation Reshape
        network = tf.reshape(network, [-1, 29, 128])  # [batch, 29, 128]
        
        # Validate LSTM output reshape
        lstm_elements = tf.shape(network)[0] * 29 * 128
        tf.debugging.assert_equal(tf.reduce_prod(tf.shape(network)), lstm_elements,
                                message="LSTM output reshape error")

        # Previous Weights Validation
        tf.debugging.assert_shapes([
            (self.previous_w, ('batch', self._rows)),  # Changed to dynamic rows
            (network, ('batch', self._rows, 128))
        ], message="Previous weights shape mismatch")

        previous_w = tf.expand_dims(self.previous_w, -1)  # [batch, rows, 1]
        network = tf.concat([network, previous_w], axis=-1)  # [batch, rows, 129]

        # Flatten
        network = tf.reshape(network, [-1, self._rows * 129])  # [batch, rows*129]

        # Layer Normalisation
        network = tf.keras.layers.LayerNormalization()(network)

        # Final Output Layer
        network = tflearn.fully_connected(
            network,
            self._rows + 1,
            activation='softmax'
        )

        # Enhanced Validation
        tf.debugging.assert_shapes([
            (network, ('batch', self._rows + 1)),
            (self.previous_w, ('batch', self._rows))
        ], message="Output/previous_w dimension mismatch")
        
        return network

def allint(l):
    return [int(i) for i in l]

