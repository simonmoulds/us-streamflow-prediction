#!/usr/bin/env python3

import os
import pickle
import pandas as pd
import xarray as xr
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from ruamel.yaml import YAML
from tqdm import tqdm

from neuralhydrology.evaluation import metrics
from neuralhydrology.nh_run import start_run, eval_run

# TESTING
nh_config_filename = 'results/month/basins.yml'
output_dir = 'results/month/lstm'

# # Get command line arguments
# nh_config_filename = sys.argv[1]
# output_dir = sys.argv[2]

# Load neuralhydrology configuration
yaml = YAML() #typ = 'safe')
nh_config = yaml.load(Path(nh_config_filename))
base_run_dir = nh_config['run_dir']
with open(os.path.join(base_run_dir, 'LATEST'), 'r') as f:
    run_dir = f.read()

# Evaluate run using the most recent run directory [NB by default this will use the last epoch]
eval_run(Path(run_dir), period = 'test')

# Unpack time series output
try:
    os.makedirs(output_dir)
except FileExistsError:
    pass

n_epochs = nh_config['epochs']
epoch_dirname = 'model_epoch' + str(n_epochs).zfill(3)
with open(os.path.join(run_dir, 'test', epoch_dirname, 'test_results.p'), 'rb') as fp:
    results = pickle.load(fp)

basins = [basin for basin in results.keys()]
n_basins = len(basins)

freq = list(set(sum([list(results[basin].keys()) for basin in basins], [])))
if len(freq) > 1:
    raise ValueError
freq = freq[0]

for i in tqdm(range(n_basins)):
    basin = basins[i]
    df = results[basin][freq]['xr'].to_dataframe()
    df.insert(0, "ID", basin)
    df = df.reset_index()
    df = df.drop('time_step', axis = 1)
    rowdata_df = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_to_dataset(
        rowdata_df,
        root_path = os.path.join(output_dir, 'test'),
        partition_cols = ['ID']
    )
