from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
from portopti.learn.rollingtrainer import RollingTrainer
import logging
import time


class Trader:
    def __init__(self, waiting_period, config, total_steps, net_dir, agent=None, initial_investment=1.0, agent_type="nn"):
        """
        @:param agent_type: string, could be nn or traditional
        @:param agent: the traditional agent object, if the agent_type is traditional
        """
        self._steps = 0
        self._total_steps = total_steps
        self._period = waiting_period
        self._agent_type = agent_type

        if agent_type == "traditional":
            config["input"]["feature_number"] = 1
            config["input"]["norm_method"] = "relative"
            self._norm_method = "relative"
        elif agent_type == "nn":
            self._rolling_trainer = RollingTrainer(config, net_dir, agent=agent)
            self._stock_name_list = self._rolling_trainer.stock_list
            self._norm_method = config["input"]["norm_method"]
            if not agent:
                agent = self._rolling_trainer.agent
        else:
            raise ValueError()
        self._agent = agent

        self._total_capital = initial_investment
        self._window_size = config["input"]["window_size"]
        self._total_stock = config["input"]["total_stock"]
        self._commission_rate = config["trading"]["trading_consumption"]
        self._fake_ratio = config["input"]["fake_ratio"]
        self._asset_vector = np.zeros(self._total_stock+1)

        self._last_omega = np.zeros((self._total_stock+1,))
        self._last_omega[0] = 1.0

        if self.__class__.__name__=="BackTest":
            self._logging_data_frame = None
        self._current_error_state = 'S000'
        self._current_error_info = ''

    def _initialize_logging_data_frame(self, initial_investment):
        logging_dict = {'Total Asset': initial_investment, '$': 10000}
        for stock in self._stock_name_list:
            logging_dict[stock] = 0
        self._logging_data_frame = pd.DataFrame(logging_dict, index=pd.to_datetime([time.time()], unit='s'))

    def generate_history_matrix(self):
        """override this method to generate the input of agent
        """
        pass

    def finish_trading(self):
        pass

    # add trading data into the pandas data frame
    def _log_trading_info(self, time, omega):
        time_index = pd.to_datetime([time], unit='s')
        if self._steps > 0:
            logging_dict = {'Total Asset': self._total_capital, '$': omega[0, 0]}
            for i in range(len(self._stock_name_list)):
                logging_dict[self._stock_name_list[i]] = omega[0, i + 1]
            new_data_frame = pd.DataFrame(logging_dict, index=time_index)
            self._logging_data_frame = self._logging_data_frame.append(new_data_frame)

    def trade_by_strategy(self, omega):
        """execute the trading to the position, represented by the portfolio vector w
        """
        pass

    def rolling_train(self):
        """
        execute rolling train
        """
        pass

    def __trade_body(self):
        self._current_error_state = 'S000'
        starttime = time.time()
        omega = self._agent.decide_by_history(self.generate_history_matrix(),
                                              self._last_omega.copy())
        self.trade_by_strategy(omega)
        if self._agent_type == "nn":
            self.rolling_train()
        if not self.__class__.__name__=="BackTest":
            self._last_omega = omega.copy()
        logging.info('total assets are $%3f' % self._total_capital)
        logging.debug("="*30)
        trading_time = time.time() - starttime
        if trading_time < self._period:
            logging.info("sleep for %s seconds" % (self._period - trading_time))
        self._steps += 1
        return self._period - trading_time

    def start_trading(self):
        try:
            if not self.__class__.__name__=="BackTest":
                current = int(time.time())
                wait = self._period - (current%self._period)
                logging.info("sleep for %s seconds" % wait)
                time.sleep(wait+2)

                while self._steps < self._total_steps:
                    sleeptime = self.__trade_body()
                    time.sleep(sleeptime)
            else:
                while self._steps < self._total_steps:
                    self.__trade_body()
        finally:
            if self._agent_type=="nn":
                self._agent.recycle()
            self.finish_trading()
