#!/usr/bin/env python3

import os
import sys
import shutil
import datetime
import re
from pathlib import Path
from tqdm import tqdm

import pyarrow
import numpy as np
import pandas as pd
import xarray as xr
from ruamel.yaml import YAML

from neuralhydrology.datasetzoo.camelsus import load_camels_us_attributes, load_camels_us_discharge, load_camels_us_forcings

# # Get command line arguments
# # config = sys.argv[1]
# outputroot = sys.argv[1]

# # For testing:
# config = 'config/config.yml'
outputroot = 'results'
# outputdir = os.path.join(os.getcwd(), outputroot)

data_dir = os.path.join('resources', 'basin_dataset_public_v1p2')

# ##################################### #
# 1 - Time series data                  #
# ##################################### #

def _load_basin_data(data_dir: Path, basin: str, forcings: list) -> pd.DataFrame:
    """Load input and output data from text files."""
    # get forcings
    dfs = []
    for forcing in forcings:
        df, area = load_camels_us_forcings(data_dir, basin, forcing)
        # rename columns
        if len(forcings) > 1:
            df = df.rename(columns={col: f"{col}_{forcing}" for col in df.columns})
        dfs.append(df)
    df = pd.concat(dfs, axis=1)

    # add discharge
    df['QObs(mm/d)'] = load_camels_us_discharge(data_dir, basin, area)

    # replace invalid discharge values by NaNs
    qobs_cols = [col for col in df.columns if "qobs" in col.lower()]
    for col in qobs_cols:
        df.loc[df[col] < 0, col] = np.nan

    return df

basin_metadata = pd.read_table(
    os.path.join(data_dir, 'basin_metadata', 'basin_physical_characteristics.txt'),
    sep = '\s+',
    header = 0,
    names = ['basin_huc', 'basin_id', 'area', 'elevation', 'slope', 'frac_forest'],
    dtype = {
        'basin_huc' : str,
        'basin_id' : str,
        'area' : float,
        'elevation' : float,
        'slope' : float,
        'frac_forest' : float
    })

for aggr in ['day', 'month', 'season']:
    try:
        os.makedirs(f'results/{aggr}/time_series')
    except FileExistsError:
        pass

forcings = ['daymet', 'maurer', 'nldas']
basins = list(basin_metadata['basin_id'])
for i in tqdm(range(len(basins))):
    basin = basins[i]
    # Note that discharge is normlized by area in this function
    try:
        # 02108000 doesn't work because maurer data is missing time columns
        df = _load_basin_data(Path(data_dir), basin, forcings)
    except AttributeError:
        continue
    ds = xr.Dataset.from_dataframe(df)
    # Remove Year/Mnth/Day variables
    varnames = list(ds.keys())
    regobj = re.compile('(Year|Mnth|Day|Hr)_(daymet|maurer|nldas)')
    varnames = [var for var in varnames if not (regobj.match(var))]
    ds_target = ds['QObs(mm/d)']
    ds_forcings = ds[varnames]
    # Day [i.e. as in original]
    ds_day = xr.merge([ds_target, ds_forcings])
    # Month
    ds_forcings_month = ds_forcings.resample(date = '1M').mean()
    ds_target_month_mean = ds_target.resample(date = '1M').mean().rename("QObs(mm/d)_mean")
    ds_target_month_max = ds_target.resample(date = '1M').max().rename("QObs(mm/d)_max")
    ds_target_month_min = ds_target.resample(date = '1M').max().rename("QObs(mm/d)_min")
    ds_month = xr.merge([ds_target_month_mean, ds_target_month_max, ds_target_month_min, ds_forcings_month])
    # Season
    ds_forcings_season = ds_forcings_month.resample(date = 'QS-DEC').mean()
    ds_target_season_mean = ds_target.resample(date = 'QS-DEC').mean().rename("QObs(mm/d)_mean")
    ds_target_season_max = ds_target.resample(date = 'QS-DEC').max().rename("QObs(mm/d)_max")
    ds_target_season_min = ds_target.resample(date = 'QS-DEC').min().rename("QObs(mm/d)_min")
    ds_season = xr.merge([ds_target_season_mean, ds_target_season_max, ds_target_season_min, ds_forcings_season])
    # Rename variables
    def make_valid_varnames(ds):
        varnames = [var for var in ds.variables]
        # (i) Replace brackets with underscore
        varnames1 = [re.sub(r'\(|\)|\/', '_', var) for var in varnames]
        # (ii) Replace consecutive underscores with one underscore
        varnames2 = [re.sub(r'_{2,}', '_', var) for var in varnames1]
        # (iii) Remove leading/trailing underscores
        varnames3 = [re.sub(r'^_|_$', '', var) for var in varnames2]
        ds = ds.rename({oldvar : newvar for oldvar, newvar in zip(varnames, varnames3)})
        return ds
    ds_day = make_valid_varnames(ds_day)
    ds_month = make_valid_varnames(ds_month)
    ds_season = make_valid_varnames(ds_season)
    # Write data to file
    ds_day.to_netcdf(os.path.join('results', 'day', 'time_series', basin + '.nc'))
    ds_month.to_netcdf(os.path.join('results', 'month', 'time_series', basin + '.nc'))
    ds_season.to_netcdf(os.path.join('results', 'season', 'time_series', basin + '.nc'))

# ##################################### #
# 2 - Catchment attributes              #
# ##################################### #

for time_period in ['day', 'month', 'season']:
    try:
        os.makedirs(os.path.join('results', time_period, 'attributes'))
    except FileExistsError:
        pass

    for attr in ['clim', 'geol', 'hydro', 'name', 'soil', 'topo', 'vege']:
        src = os.path.join(data_dir, '..', f'camels_{attr}.txt')
        dst = os.path.join('results', time_period, 'attributes', f'camels_{attr}.txt')
        shutil.copy(src, dst)

# ##################################### #
# 3 - NH configuration                  #
# ##################################### #

# TODO ensure this matches canonical US CAMELS LSTM paper

# climatic_attributes = ['p_mean', 'pet_mean', 'aridity', 'p_seasonality', 'frac_snow', 'high_prec_freq', 'high_prec_dur', 'low_prec_freq', 'low_prec_dur']
# human_influence_attributes = []
# hydrogeology_attributes = ['inter_high_perc', 'inter_mod_perc', 'inter_low_perc', 'frac_high_perc', 'frac_mod_perc', 'frac_low_perc', 'no_gw_perc', 'low_nsig_perc', 'nsig_low_perc']
# hydrologic_attributes = []
# landcover_attributes = ['dwood_perc', 'ewood_perc', 'urban_perc']
# soil_attributes = ['sand_perc', 'silt_perc', 'clay_perc', 'porosity_hypres', 'conductivity_hypres', 'soil_depth_pelletier_50']
# topographic_attributes = ['gauge_lat', 'gauge_lon', 'gauge_elev', 'area', 'elev_10', 'elev_50', 'elev_90']

# static_attributes = climatic_attributes + \
#     human_influence_attributes + \
#     hydrogeology_attributes + \
#     hydrologic_attributes + \
#     landcover_attributes + \
#     soil_attributes + \
#     topographic_attributes

# dynamic_inputs = ['P', 'T', 'EA', 'AMV']
# # static_attributes = ['gauge_lat', 'gauge_lon', 'p_mean', 'pet_mean', 'area'] #, 'q_mean']#, 'frac_snow', 'high_prec_freq', 'high_prec_dur', 'low_prec_freq', 'low_prec_dur']

# yaml = YAML() #typ = 'safe')
# cfg = yaml.load(Path('resources/nh-config-template.yml'))
# cfg['experiment_name'] = 'cudalstm_' + str(n_stations) + '_basins_' + str(aggr_period)
# cfg['run_dir'] = rundir
# cfg['train_basin_file'] = os.path.join(outputdir, "basins.txt")
# cfg['validation_basin_file'] = os.path.join(outputdir, "basins.txt")
# cfg['test_basin_file'] = os.path.join(outputdir, "basins.txt")
# cfg['train_start_date'] = ["01/12/1961", "01/12/1992"] #"01/12/1961"
# cfg['train_end_date'] = ["01/12/1990", "01/12/2006"] #"01/12/1990", #"01/12/2006"
# cfg['validation_start_date'] = "01/12/1991" #"01/12/1961"
# cfg['validation_end_date'] = "01/12/1991"
# cfg['test_start_date'] = "01/12/1961"
# cfg['test_end_date']= "01/12/2006"
# cfg['device'] = "cpu" # "cuda:0"
# cfg['validate_every'] = int(3)
# cfg['validate_n_random_basins'] = int(1)
# cfg['metrics'] = ["MSE"]
# cfg['save_validation_results'] = True
# cfg['model'] = "cudalstm" #"cudalstm" # "ealstm"
# cfg['head'] = "regression"
# cfg['output_activation'] = "linear"
# cfg['hidden_size'] = int(64) #int(128) #int(20) #int(64) #int(20)
# cfg['initial_forget_bias'] = int(3)
# cfg['output_dropout'] = 0.4
# cfg['optimizer'] = "Adam"
# cfg['loss'] = "MSE"
# cfg['learning_rate'] = {0: 1e-2, 30: 5e-3, 40: 1e-3}
# cfg['batch_size'] = int(128) #int(256)
# cfg['epochs'] = int(50) #int(50) #int(500) #int(150) #int(75)
# cfg['clip_gradient_norm'] = int(1)
# cfg['predict_last_n'] = int(1)
# cfg['seq_length'] = int(4) #int(4) #int(8)
# cfg['num_workers'] = int(8)
# cfg['log_interval'] = int(5)
# cfg['log_tensorboard'] = True
# cfg['log_n_figures'] = int(0)
# cfg['save_weights_every'] = int(1)
# cfg['dataset'] = "generic"
# cfg['data_dir'] = outputdir
# cfg['dynamic_inputs'] = dynamic_inputs
# cfg['target_variables'] = ["Q95"]
# cfg['clip_targets_to_zero'] = ["Q95"]
# cfg['static_attributes'] = static_attributes

# for time_period in ['day', 'month', 'season']:
#     conf_filename = os.path.join('results', time_period, 'basins.yml')
#     with open(conf_filename, 'wb') as f:
#         yaml.dump(cfg, f)

# # ##################################### #
# # 4 - Basins list                       #
# # ##################################### #

# basins = list(basin_metadata['basin_id'])
# for time_period in ['day', 'month', 'season']:
#     basin_filename = os.path.join('results', time_period, 'basins.txt')
#     with open(basin_filename, 'w') as f:
#         for stn in station_ids:
#             f.write(str(stn))
#             f.write('\n')
