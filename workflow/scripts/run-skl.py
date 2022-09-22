#!/usr/bin/env python3

import os
import sys
import re
import shutil
import pickle
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray
import datetime
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Union
from ruamel.yaml import YAML
from tqdm import tqdm

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

# import matplotlib.pyplot as plt
# import torch
# from neuralhydrology.evaluation import metrics
# from neuralhydrology.nh_run import start_run, eval_run
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import MSELoss
from torch.nn.init import xavier_uniform_

# TabNet
from pytorch_tabnet.tab_model import TabNetRegressor

# Some helpers from neuralhydrology
from neuralhydrology.utils.config import Config
from neuralhydrology.datautils.utils import load_basin_file
from neuralhydrology.datasetzoo.genericdataset import load_timeseries, load_attributes

# Dataset definition
class DCPDataset(Dataset):
    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        self.cfg = cfg
        self.is_train = is_train

        if period not in ["train", "validation", "test"]:
            raise ValueError("'period' must be one of 'train', 'validation' or 'test' ")
        else:
            self.period = period

        if period in ["validation", "test"]:
            if not scaler:
                raise ValueError("During evaluation of validation or test period, scaler dictionary has to be passed")

        data_dir = cfg.data_dir
        basins = Path(getattr(cfg, f'{period}_basin_file'))
        with basins.open('r') as f:
            basins = f.read().splitlines()

        start_dates = getattr(cfg, f'{period}_start_date')
        end_dates = getattr(cfg, f'{period}_end_date')
        if not isinstance(start_dates, list):
            start_dates = [start_dates]
            end_dates = [end_dates]

        # Read dynamic input data
        data_list = []
        for basin in basins:
            df = load_timeseries(data_dir, basin) # TESTING
            # create xarray data set for each period slice of the specific basin
            for i, (start_date, end_date) in enumerate(zip(start_dates, end_dates)):
                df_sub = df[(df.index >= start_date) & (df.index <= end_date)]
                xr = xarray.Dataset.from_dataframe(df_sub)
                basin_str = basin if i == 0 else f"{basin}_period{i}"
                xr = xr.assign_coords({'basin': basin_str})
                data_list.append(xr.astype(np.float32))

        # create one large dataset that has two coordinates: datetime and basin
        xr = xarray.concat(data_list, dim="basin")

        # Read attributes
        attr = load_attributes(data_dir, basins)
        attr = attr.reset_index()
        attr = attr.set_index('gauge_id')
        # static_inputs = cfg.static_attributes
        attr = attr[cfg.static_attributes]

        self.scaler = scaler
        if self.is_train:
            self.scaler['xarray_feature_center'] = xr.mean()
            self.scaler['xarray_feature_scale'] = xr.std()
            self.scaler['attribute_means'] = attr.mean()
            self.scaler['attribute_stds'] = attr.std()
            self._dump_scaler()

        attr = (attr - self.scaler['attribute_means']) / self.scaler['attribute_stds']
        xr = (xr - self.scaler['xarray_feature_center']) / self.scaler['xarray_feature_scale']

        # merge dynamic and static inputs
        attr = attr.reset_index()
        dyn_inputs = xr[cfg.dynamic_inputs].to_dataframe()
        dyn_inputs = dyn_inputs.reset_index()
        dyn_inputs['gauge_id'] = [re.sub('_period[0-9]+', '', basin) for basin in dyn_inputs['basin']]
        # dyn_inputs = dyn_inputs.set_index(['gauge_id', 'basin', 'date'])
        inputs = pd.merge(
            dyn_inputs,
            attr,
            how = 'left',
            on = 'gauge_id'
        )
        inputs = inputs.reset_index()
        inputs = inputs.drop('gauge_id', axis = 1)
        self.index = inputs[['date', 'basin']]
        inputs = inputs.set_index(['date', 'basin'])

        self.y = xr[self.cfg.target_variables].to_dataframe().values
        self.X = inputs.values
        # Ensure floats
        self.y = self.y.astype('float32')
        self.X = self.X.astype('float32')
        # Validate (check for NaN)
        target_idx = np.array([all(~np.isnan(self.y[i,:])) for i in range(len(self.y))])
        input_idx = np.array([all(~np.isnan(self.X[i,:])) for i in range(len(self.X))])
        nan_idx = target_idx & input_idx
        self.y = self.y[nan_idx,:]
        self.X = self.X[nan_idx,:]
        self.index = self.index[nan_idx]

    def _dump_scaler(self):
        # dump scaler dictionary into run directory for inference
        scaler = defaultdict(dict)
        for key, value in self.scaler.items():
            if isinstance(value, pd.Series) or isinstance(value, xarray.Dataset):
                scaler[key] = value.to_dict()
            else:
                raise RuntimeError(f"Unknown datatype for scaler: {key}. Supported are pd.Series and xarray.Dataset")
        # file_path = self.cfg.train_dir / "train_data_scaler.yml"
        file_path = self.cfg.data_dir / "train_data_scaler.yml"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w") as fp:
            yaml = YAML()
            yaml.dump(dict(scaler), fp)

    def __len__(self):
        return(len(self.X))

    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]


# # Get command line arguments
# nh_config = sys.argv[1]
# outputdir = sys.argv[2]

# # Load configuration
# cfg = Config(Path(nh_config))
# train_dl, test_dl = prepare_data(cfg)
# model = MLP(38)                  # not sure where 4 comes from?
# train_model(train_dl, model)
# acc = evaluate_model(test_dl, model) # NaN problem

# For handling NaN, see validate_samples in basedataset.py
# For now, just remove all data points that have an NaN

def run_tabular_regression(cfg: Config, model_type: str):
    # Get train, test datasets
    target_variable = cfg.target_variables
    if len(target_variable) != 1:
        raise ValueError("Currently only one target variable is supported")
    else:
        target_variable = target_variable[0]
    train = DCPDataset(cfg, is_train = True, period = 'train')
    scaler_file_path = cfg.data_dir / "train_data_scaler.yml"
    with open(scaler_file_path) as f:
        yaml = YAML()
        scaler_dict = dict(yaml.load(f))
        scaler_dict['xarray_feature_center'] = xarray.Dataset.from_dict(scaler_dict['xarray_feature_center'])
        scaler_dict['xarray_feature_scale'] = xarray.Dataset.from_dict(scaler_dict['xarray_feature_scale'])
        scaler_dict['attribute_means'] = pd.Series(scaler_dict['attribute_means'])
        scaler_dict['attribute_stds'] = pd.Series(scaler_dict['attribute_stds'])
    test = DCPDataset(cfg, is_train = False, period = 'test', scaler = scaler_dict)
    # Fit model using k-fold cross validation, taking average across predictions
    kf = KFold(n_splits = 5, random_state = 42, shuffle = True)
    predictions_array = []
    for train_index, test_index in kf.split(train.X):
        X_train, X_valid = train.X[train_index], train.X[test_index]
        y_train, y_valid = train.y[train_index], train.y[test_index]

        if model_type in ['xgboost']:
            model = XGBRegressor(n_estimators = 500)
            model.set_params(early_stopping_rounds = 5) #, eval_metric = [r2_score])
            model.fit(X_train, y_train, verbose = False, eval_set = [(X_valid, y_valid)])

        elif model_type in ['tabnet']:
            model = TabNetRegressor(verbose = 0, seed = 42)
            model.fit(X_train, y_train, eval_set = [(X_valid, y_valid)])
        else:
            raise ValueError('`model_type` must be one of "xgboost", "tabnet"')
        predictions_array.append(model.predict(test.X))

    predictions = np.mean(predictions_array, axis = 0) #model.predict(test.X)
    output = test.index
    target_center = float(scaler_dict['xarray_feature_center'][target_variable].values)
    target_std = float(scaler_dict['xarray_feature_scale'][target_variable].values)
    output[target_variable + '_obs'] = (test.y * target_std) + target_center
    output[target_variable + '_exp'] = (predictions * target_std) + target_center
    return output

# # TESTING
# config = 'config/config.yml'
# nh_config = 'results/exp2/analysis/yr2/nh-input/basins.yml'
# aggregation_period = 'yr2'
# model_type = 'xgboost'
# outputdir = os.path.join('results/exp2/analysis/hindcast/',  model_type)

# Get command line arguments
config = sys.argv[1]
nh_config = sys.argv[2]
aggregation_period = sys.argv[3]
model_type = sys.argv[4]
outputdir = sys.argv[5]

# Load neuralhydrology configuration
yaml = YAML() #typ = 'safe')
nh_cfg = yaml.load(Path(nh_config))
cfg = yaml.load(Path(config))
aggr_period_info = [d for d in cfg['aggregation_period'] if d['name'] == aggregation_period]
if len(aggr_period_info) != 1:
    ValueError
else:
    aggr_period_info = dict(aggr_period_info[0])
lead_time = str(aggr_period_info['lead_time'])
lead_time = [int(i) for i in lead_time.split(':')]
if len(lead_time) > 1:
    lead_time = [i for i in range(lead_time[0], lead_time[1] + 1)]

# # Where to put predictions
# prediction_outputdir = os.path.join(outputdir, aggregation_period)
# # TODO this won't be needed when running from Snakefile
# try:
#     os.makedirs(prediction_outputdir)
# except FileExistsError:
#     pass

# Set up while-loop
leave_out = len(lead_time)
train_year_start = 1961
train_year_end = 1979
test_year_start = train_year_end + leave_out
max_test_year_start = 2006
df_list = []
n_runs = max_test_year_start - test_year_start
pbar = tqdm(total = n_runs)
while test_year_start <= max_test_year_start:
    cfg = Config(Path(nh_config))
    cfg.update_config({'target_variables' : ['Q95']})
    cfg.update_config(
        {'train_start_date' : '01/12/' + str(train_year_start),
         'train_end_date' : '01/12/' + str(train_year_end),
         'test_start_date' : '01/12/' + str(test_year_start),
         'test_end_date' : '01/12/' + str(test_year_start)#,
         # 'static_attributes' : [],
         # 'dynamic_inputs' : ['P', 'T', 'EA', 'AMV'] #, 'NAO']
         })
    df = run_tabular_regression(cfg, model_type = model_type)
    df = df.rename({'basin' : 'ID'}, axis = 1)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = [tm.year for tm in df['date']]
    rowdata_df = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_to_dataset(
        rowdata_df,
        root_path = os.path.join(outputdir, 'prediction'),
        partition_cols = ['ID', 'date']
    )
    train_year_end += 1
    test_year_start += 1
    pbar.update(1)

# Close progress bar
pbar.close()


# NOT USED:

# Example of fully-connected deep neural network:
# class MLP(Module):
#     def __init__(self, n_inputs):
#         super(MLP, self).__init__()
#         # Input to first hidden layer
#         self.hidden1 = Linear(n_inputs, 10)
#         xavier_uniform_(self.hidden1.weight)
#         self.act1 = Sigmoid()
#         # Input to second hidden layer
#         self.hidden2 = Linear(10, 8)
#         xavier_uniform_(self.hidden1.weight)
#         self.act2 = Sigmoid()
#         # Third hidden layer and output
#         self.hidden3 = Linear(8, 1)
#         xavier_uniform_(self.hidden3.weight)

#     def forward(self, X):
#         # Input to first hidden layer
#         X = self.hidden1(X)
#         X = self.act1(X)
#         # Second hidden layer
#         X = self.hidden2(X)
#         X = self.act2(X)
#         # Third hidden layer and output
#         X = self.hidden3(X)
#         return X

# def prepare_mlp_data(cfg: Config):
#     train = DCPDataset(cfg, is_train = True, period = 'train')
#     test = DCPDataset(cfg, is_train = False, period = 'test')
#     # Prepare data loaders
#     train_dl = DataLoader(train, batch_size = cfg.batch_size, shuffle = True)
#     test_dl = DataLoader(test, batch_size = 1024, shuffle = False)
#     return train_dl, test_dl

# def train_model(train_dl, model):
#     criterion = MSELoss()
#     # lr is learning rate, which could change by epoch (see neuralhydrology.training)
#     optimizer = SGD(model.parameters(), lr = 0.01, momentum = 0.9)
#     for epoch in range(100):
#         for i, (inputs, targets) in enumerate(train_dl):
#             # Clear the gradients
#             optimizer.zero_grad()
#             # Compute the model output
#             yhat = model(inputs)
#             # Calculate loss
#             loss = criterion(yhat, targets)
#             # Credit assignment
#             loss.backward()
#             # Update model weights
#             optimizer.step()

# def evaluate_model(test_dl, model):
#     predictions, actuals = list(), list()
#     for i, (inputs, targets) in enumerate(test_dl):
#         # Evaluate the model on the test set
#         yhat = model(inputs)
#         # Retrieve numpy array
#         yhat = yhat.detach().numpy()
#         actual = targets.numpy()
#         actual = actual.reshape((len(actual), 1))
#         # Store
#         predictions.append(yhat)
#         actuals.append(actual)
#     predictions, actuals = vstack(predictions), vstack(actuals)
#     # Calculate MSE
#     mse = mean_squared_error(actuals, predictions)
#     return mse

# def predict(row, model):
#     # Convert row to data
#     row = Tensor([row])
#     # Make prediction
#     yhat = model(row)
#     # Retrieve numpy array
#     yhat = yhat.detach().numpy()
#     return yhat
