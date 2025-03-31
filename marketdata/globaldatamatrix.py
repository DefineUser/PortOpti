from __future__ import division, absolute_import, print_function

import numpy as np
import pandas as pd
import xarray as xr
import logging
from datetime import datetime

from portopti.tools.data import create_data_array # helper functions for Xarray creation
from portopti.constants import *  # Constants like FIVE_MINUTES, DAY, etc.

class HistoryManager:
    """
    HistoryManager that loads data from a CSV.
    """
    def __init__(self, total_stock, end, volume_average_days=1, volume_forward=0, csv_path=None):
        self._total_stock = total_stock
        self.__storage_period = FIVE_MINUTES  # kept for period adjustment purposes
        self.__volume_forward = volume_forward
        self.__volume_average_days = volume_average_days
        self.__stocks = None

        if csv_path is None:
            raise ValueError("Offline mode requires a csv_path to be provided")
        self._data = pd.read_csv(csv_path)

        # Convert the CSV 'date' column:
        # If numeric, treat as UNIX timestamps; otherwise, assume DD/MM/YYYY and set dayfirst=True.
        if pd.api.types.is_numeric_dtype(self._data['date']):
            self._data['date'] = pd.to_datetime(self._data['date'], unit='s').dt.date
        else:
            self._data['date'] = pd.to_datetime(self._data['date'], dayfirst=True).dt.date

    @property
    def stocks(self):
        return self.__stocks

    def get_global_panel(self, start, end, period=300, features=('close',)):
        """
        Returns a 3D tensor as an Xarray DataArray with dimensions:
        [feature, stock, time].
        The start and end are given as UNIX timestamps; they are adjusted and converted to dates.
        """
        # Adjust start and end to be multiples of period.
        start_ts = int(start - (start % period))
        end_ts = int(end - (end % period))
        
        # Convert the adjusted timestamps to date objects for filtering.
        filter_start_date = pd.to_datetime(start_ts - self.__volume_forward - self.__volume_average_days * DAY, unit='s').date()
        filter_end_date = pd.to_datetime(end_ts - self.__volume_forward, unit='s').date()

        logging.debug("CSV date range: min=%s, max=%s", self._data['date'].min(), self._data['date'].max())
        logging.debug("Filter start date: %s, Filter end date: %s", filter_start_date, filter_end_date)

        # Select stocks (tickers) based on total volume in the CSV data.
        stocks = self.select_stocks(start_date=filter_start_date, end_date=filter_end_date)
        self.__stocks = stocks
        logging.debug("HistoryManager __stocks set to: " + str(self.__stocks))

        if len(stocks) != self._total_stock:
            raise ValueError("The length of selected stocks %d is not equal to expected %d" %
                             (len(stocks), self._total_stock))

        logging.info("Feature type list: %s" % str(features))
        self.__checkperiod(period)

        # Create a time index identical to the original code.
        time_index = pd.to_datetime(list(range(start_ts, end_ts + 1, period)), unit='s')

        # Create an empty Xarray DataArray using helper.
        data_array = create_data_array(features, stocks, time_index)
        # After populating data_array
        assert data_array.feature.size == len(features), \
        f"Feature dimension mismatch: {data_array.feature.size} vs {len(features)}"

        # For each stock and each feature, aggregate the CSV data by day.
        for stock in stocks:
            # Raw data validation
            stock_data = self._data[self._data['stock'] == stock]
            if stock_data.empty:
                raise ValueError(f"No data found for stock {stock}")
                
            # Check for invalid close prices
            if (stock_data['close'] <= 0).any():
                logging.error(f"Invalid close prices (<=0) found in {stock}")
                stock_data = stock_data[stock_data['close'] > 0].copy()
                
            df_stock = stock_data.sort_values('date').reset_index(drop=True)
            
            # 1. CALCULATE RETURNS SAFELY
            # Compute returns with numerical stability
            df_stock['returns'] = (df_stock['close'] - df_stock['close'].shift(1)) / (df_stock['close'].shift(1) + 1e-8)
            df_stock['returns'] = df_stock['returns'].fillna(0)  # Fill initial NaN
            
            # Handle infinite/overflow values
            df_stock['returns'] = np.clip(df_stock['returns'], -1, 1)  # Cap extreme returns at ±100%
            
            # 2. TECHNICAL INDICATORS
            # Volume normalisation
            df_stock['volume_ma30'] = df_stock['volume'].rolling(30, min_periods=1).mean()
            df_stock['norm_volume'] = df_stock['volume'] / (df_stock['volume_ma30'] + 1e-8)
            
            # Price relatives
            df_stock['rel_high'] = (df_stock['high'] - df_stock['open']) / (df_stock['open'] + 1e-8)
            df_stock['rel_low'] = (df_stock['low'] - df_stock['open']) / (df_stock['open'] + 1e-8)
            
            # Moving averages
            df_stock['MA5'] = df_stock['close'].rolling(5, min_periods=1).mean()
            df_stock['MA21'] = df_stock['close'].rolling(21, min_periods=1).mean()
            df_stock['MA_ratio'] = df_stock['MA5'] / (df_stock['MA21'] + 1e-8)
            
            # Volatility calculation
            df_stock['volatility'] = df_stock['returns'].rolling(21, min_periods=1).std().fillna(0)
            
            # 3. TEMPORAL ALIGNMENT
            df_stock = df_stock.set_index('date')
            df_stock = df_stock.reindex(time_index.date, method='ffill').ffill().bfill()
            
            # 4. FEATURE POPULATION WITH VALIDATION
            for f_idx, feature in enumerate(features):
                if feature == "returns":
                    vals = df_stock['returns'].shift(1).fillna(0).values
                elif feature == "norm_volume":
                    vals = df_stock['norm_volume'].fillna(0).values
                elif feature == "MA_ratio":
                    vals = df_stock['MA_ratio'].replace([np.inf, -np.inf], 0).fillna(0).values
                elif feature == "volatility":
                    vals = df_stock['volatility'].fillna(0).values
                elif feature == "close":
                    vals = df_stock['close'].values
                elif feature == "rel_high":
                    vals = df_stock['rel_high'].fillna(0).values
                elif feature == "rel_low":
                    vals = df_stock['rel_low'].fillna(0).values
                else:
                    valid_features = {"returns", "norm_volume", "MA_ratio", "volatility", "close", "rel_high", "rel_low"}
                    raise ValueError(f"Unsupported feature: {feature}. Valid options: {valid_features}")

                # Clean values
                vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
                vals = np.clip(vals, -1, 1)  # For price relatives
                
                # Populate data array
                for date_val, value in zip(df_stock.index, vals):
                    ts = pd.to_datetime(date_val)
                    if ts in time_index:
                        t_idx = time_index.get_loc(ts)
                        c_idx = stocks.index(stock)
                        data_array[f_idx, c_idx, t_idx] = value

        # 5. POST-PROCESSING CHECKS
        # Validate returns
        returns = data_array.sel(feature="returns").values
        if not np.all(np.isfinite(returns)):
            logging.warning("NaN/Inf found in returns, replacing with 0")
            returns = np.nan_to_num(returns)
            data_array.loc[dict(feature="returns")] = returns
            
        # Validate volatility
        volatility = data_array.sel(feature="volatility").values
        if (volatility < 0).any():
            logging.error("Negative volatility values detected")
            volatility = np.abs(volatility)
            data_array.loc[dict(feature="volatility")] = volatility

        # 6. NORMALISATION
        def safe_rolling_normalise(da):
            rolling_mean = da.rolling(time=21, min_periods=1).mean()
            rolling_std = da.rolling(time=21, min_periods=1).std()
            normalized = (da - rolling_mean) / (rolling_std + 1e-8)
            return normalized.where(np.isfinite(normalized), 0)

        data_array = data_array.groupby('feature').apply(safe_rolling_normalise)
        
        return data_array

    def select_stocks(self, start_date, end_date):
        """
        Select the top stocks based on total trading volume
        over the period from start_date to end_date.
        """
        df = self._data.copy()
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        stock_volume = df.groupby('stock')['volume'].sum()
        top_stocks = stock_volume.nlargest(self._total_stock).index.tolist()
        if len(top_stocks) != self._total_stock:
            logging.error("Selected stocks count {} is less than expected {}".format(len(top_stocks), self._total_stock))
        logging.debug("Selected stocks: " + str(top_stocks))
        return top_stocks

    def __checkperiod(self, period):
        if period in (FIVE_MINUTES, FIFTEEN_MINUTES, HALF_HOUR, TWO_HOUR, FOUR_HOUR, DAY):
            return
        else:
            raise ValueError('Period must be one of: 5min, 15min, 30min, 2hr, 4hr, or a day')
