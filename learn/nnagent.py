from __future__ import absolute_import, print_function, division
import tflearn
import tensorflow as tf
import numpy as np
from portopti.constants import *
import portopti.learn.network as network

class NNAgent:
    def __init__(self, config, restore_dir=None, device="cpu"):
        self.__config = config
        self.__total_stock = config["input"]["total_stock"]
        self.__net = network.CNN(config["input"]["feature_number"],
                                 self.__total_stock,
                                 config["input"]["window_size"],
                                 config["layers"],
                                 device=device)
        self.__global_step = tf.Variable(0, trainable=False)
        self.__train_operation = None
        self.__y = tf.placeholder(tf.float32, shape=[None,
                                                     self.__config["input"]["feature_number"],
                                                     self.__total_stock])
        self.__future_price = tf.concat([tf.ones([self.__net.input_num, 1]),
                                       self.__y[:, 0, :]], 1)
        self.__future_omega = (self.__future_price * self.__net.output) /\
                              tf.reduce_sum(self.__future_price * self.__net.output, axis=1)[:, None]
        # tf.assert_equal(tf.reduce_sum(self.__future_omega, axis=1), tf.constant(1.0))
        self.__commission_ratio = self.__config["trading"]["trading_consumption"]
        self.__pv_vector = tf.reduce_sum(self.__net.output * self.__future_price, reduction_indices=[1]) *\
                           (tf.concat([tf.ones(1), self.__pure_pc()], axis=0))
        self.__log_mean_free = tf.reduce_mean(tf.log(tf.reduce_sum(self.__net.output * self.__future_price,
                                                                   reduction_indices=[1])))
        self.__portfolio_value = tf.reduce_prod(self.__pv_vector)
        self.__mean = tf.reduce_mean(self.__pv_vector)
        self.__log_mean = tf.reduce_mean(tf.log(self.__pv_vector))
        self.__standard_deviation = tf.sqrt(tf.reduce_mean((self.__pv_vector - self.__mean) ** 2))
        self.__sharp_ratio = (self.__mean - 1) / self.__standard_deviation
        self.__loss = self.__set_loss_function()
        self.__train_operation = self.init_train(learning_rate=self.__config["training"]["learning_rate"],
                                                 decay_steps=self.__config["training"]["decay_steps"],
                                                 decay_rate=self.__config["training"]["decay_rate"],
                                                 training_method=self.__config["training"]["training_method"])
        self.__saver = tf.train.Saver()
        if restore_dir:
            self.__saver.restore(self.__net.session, restore_dir)
        else:
            self.__net.session.run(tf.global_variables_initializer())
            

    @property
    def session(self):
        return self.__net.session

    @property
    def pv_vector(self):
        return self.__pv_vector

    @property
    def standard_deviation(self):
        return self.__standard_deviation

    @property
    def portfolio_weights(self):
        return self.__net.output

    @property
    def sharp_ratio(self):
        return self.__sharp_ratio

    @property
    def log_mean(self):
        return self.__log_mean

    @property
    def log_mean_free(self):
        return self.__log_mean_free

    @property
    def portfolio_value(self):
        return self.__portfolio_value

    @property
    def loss(self):
        return self.__loss

    @property
    def layers_dict(self):
        return self.__net.layers_dict

    def recycle(self):
        tf.reset_default_graph()
        self.__net.session.close()

    def __set_loss_function(self):

        # New MVO-inspired hybrid loss function
        def mvo_hybrid():
            # 1. Numerical Stability Foundation
            with tf.name_scope("StabilityGuards"):
                pv_vector = tf.debugging.check_numerics(self.pv_vector, "Invalid PV values")
                net_output = tf.debugging.check_numerics(self.__net.output, "Invalid network output")
                previous_w = tf.debugging.check_numerics(self.__net.previous_w, "Invalid previous weights")

            # 2. Portfolio Value Safeguards
            pv_safe = tf.clip_by_value(pv_vector, 1e-8, 1e8)  # Prevent extreme values
            log_return = tf.reduce_mean(tf.math.log(pv_safe + 1e-10))

            # 3. Stable Returns Calculation
            with tf.name_scope("Returns"):
                pv_prev = pv_safe[:-1]
                pv_next = pv_safe[1:]
                returns = (pv_prev - pv_next) / (pv_next + 1e-10)
                returns = tf.clip_by_value(returns, -1.0, 1.0)  # Constrain return range

            # 4. Variance Calculation with Initialization
            with tf.name_scope("Variance"):
                portfolio_variance = tf.math.reduce_variance(returns) + 1e-8  # Prevent zero variance

            # 5. Weight Management
            with tf.name_scope("Weights"):
                weights = tf.clip_by_value(net_output[:, 1:], 1e-6, 1.0)
                valid_weights = weights / tf.reduce_sum(weights, axis=1, keepdims=True)

            # 6. Turnover Stability
            with tf.name_scope("Turnover"):
                prev_w_safe = tf.clip_by_value(previous_w, 1e-6, 1.0)
                turnover = tf.reduce_mean(tf.abs(valid_weights - prev_w_safe))

            # 7. Final Loss Construction
            loss = -log_return + \
                self.__config["training"]["lambda_var"] * portfolio_variance + \
                self.__config["training"]["lambda_turnover"] * turnover
            
            return tf.clip_by_value(loss, -1e6, 1e6)  # Final safeguard

        loss_function = mvo_hybrid
        if self.__config["training"]["loss_function"] == "mvo_hybrid":
            loss_function = mvo_hybrid

        loss_tensor = loss_function()
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if regularization_losses:
            for regularization_loss in regularization_losses:
                loss_tensor += regularization_loss
        return loss_tensor

    def init_train(self, learning_rate, decay_steps, decay_rate, training_method):
        learning_rate = tf.train.exponential_decay(learning_rate, self.__global_step,
                                                   decay_steps, decay_rate, staircase=True)
        if training_method == 'GradientDescent':
            train_step = tf.train.GradientDescentOptimizer(learning_rate).\
                         minimize(self.__loss, global_step=self.__global_step)
        elif training_method == 'Adam':
            train_step = tf.train.AdamOptimizer(learning_rate).\
                         minimize(self.__loss, global_step=self.__global_step)
        elif training_method == 'RMSProp':
            train_step = tf.train.RMSPropOptimizer(learning_rate).\
                         minimize(self.__loss, global_step=self.__global_step)
        else:
            raise ValueError()
        return train_step

    def train(self, x, y, last_w, setw):
        assert not np.any(np.isnan(last_w)), "NaN in last_w before training"
        
        # Scale inputs to [0,1]
        x = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-10)
        y = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-10)
        
        tflearn.is_training(True, self.__net.session)
        self.evaluate_tensors(x, y, last_w, setw, [self.__train_operation])

    def evaluate_tensors(self, x, y, last_w, setw, tensors):
        """
        :param x:
        :param y:
        :param last_w:
        :param setw: a function, pass the output w to it to fill the PVM
        :param tensors:
        :return:
        """
        tensors = list(tensors)
        tensors.append(self.__net.output)
        assert not np.any(np.isnan(x))
        assert not np.any(np.isnan(y))
        assert not np.any(np.isnan(last_w)),\
            "the last_w is {}".format(last_w)
        results = self.__net.session.run(tensors,
                                         feed_dict={self.__net.input_tensor: x,
                                                    self.__y: y,
                                                    self.__net.previous_w: last_w,
                                                    self.__net.input_num: x.shape[0]})
        setw(results[-1][:, 1:])
        return results[:-1]

    # save the variables path including file name
    def save_model(self, path):
        self.__saver.save(self.__net.session, path)

    # consumption vector (on each periods)
    def __pure_pc(self):
        c = self.__commission_ratio
        w_t = self.__future_omega[:self.__net.input_num-1]  # rebalanced
        w_t1 = self.__net.output[1:self.__net.input_num]
        mu = 1 - tf.reduce_sum(tf.abs(w_t1[:, 1:]-w_t[:, 1:]), axis=1)*c
        """
        mu = 1-3*c+c**2

        def recurse(mu0):
            factor1 = 1/(1 - c*w_t1[:, 0])
            if isinstance(mu0, float):
                mu0 = mu0
            else:
                mu0 = mu0[:, None]
            factor2 = 1 - c*w_t[:, 0] - (2*c - c**2)*tf.reduce_sum(
                tf.nn.relu(w_t[:, 1:] - mu0 * w_t1[:, 1:]), axis=1)
            return factor1*factor2

        for i in range(20):
            mu = recurse(mu)
        """
        return mu

    # the history is a 3d matrix, return a asset vector
    def decide_by_history(self, history, last_w):
        assert isinstance(history, np.ndarray),\
            "the history should be a numpy array, not %s" % type(history)
        assert not np.any(np.isnan(last_w))
        assert not np.any(np.isnan(history))
        tflearn.is_training(False, self.session)
        history = history[np.newaxis, :, :, :]
        return np.squeeze(self.session.run(self.__net.output, feed_dict={self.__net.input_tensor: history,
                                                                         self.__net.previous_w: last_w[np.newaxis, 1:],
                                                                         self.__net.input_num: 1}))
