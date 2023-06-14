#!/usr/bin/env python3

import os
import sys
import shutil
import datetime
import re
import glob
import warnings
from pathlib import Path
from tqdm import tqdm

import pyarrow
import numpy as np
import pandas as pd
import xarray as xr
from ruamel.yaml import YAML, representer

from neuralhydrology.datasetzoo.camelsus import load_camels_us_attributes, load_camels_us_discharge, load_camels_us_forcings

# # Get command line arguments
# # config = sys.argv[1]
# outputroot = sys.argv[1]

# # For testing:
# config = 'config/config.yml'
# outputroot = 'results'
outputdir = os.path.join(os.getcwd(), 'results')

data_dir = os.path.join('resources', 'basin_dataset_public_v1p2')

# ##################################### #
# 1 - Basins list                       #
# ##################################### #

src = os.path.join('resources', '531_basin_list.txt')
for time_period in ['day', 'month', 'season']:
    dst = os.path.join('results', time_period)
    shutil.copy(src, dst)

with open(src, 'r') as f:
    basins = f.read().splitlines()

# ##################################### #
# 2 - Time series data                  #
# ##################################### #

def _repair_maurer_header(data_dir, basin):
    replacement_header = 'Year Mnth Day Hr\tDayl(s)\tPRCP(mm/day)\tSRAD(W/m2)\tSWE(mm)\tTmax(C)\tTmin(C)\tVp(Pa)\n'
    filepath = glob.glob('resources/basin_dataset_public_v1p2/basin_mean_forcing/maurer/*/' + basin + '_lump_maurer_forcing_leap.txt')[0]
    with open(filepath.format(basin = basin), 'r') as f:
        contents = f.readlines()

    contents[3] = replacement_header
    with open(filepath.format(basin = basin), 'w') as f:
        f.writelines(contents)


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

# Make repair to basin 02108000
_repair_maurer_header(data_dir, '02108000')
_repair_maurer_header(data_dir, '05120500')
_repair_maurer_header(data_dir, '09492400')

forcings = ['daymet', 'maurer', 'nldas']
for i in tqdm(range(len(basins))):
    basin = basins[i]
    # Note that discharge is normlized by area in this function
    df = _load_basin_data(Path(data_dir), basin, forcings)
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
    ds_forcings_month = ds_forcings.resample(date = '1MS').mean()
    ds_target_month_mean = ds_target.resample(date = '1MS').mean().rename("QObs(mm/d)_mean")
    ds_target_month_max = ds_target.resample(date = '1MS').max().rename("QObs(mm/d)_max")
    ds_target_month_min = ds_target.resample(date = '1MS').max().rename("QObs(mm/d)_min")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        ds_target_month_q95 = ds_target.resample(date = '1MS').quantile(.95).rename("QObs(mm/d)_q95")
        ds_target_month_q05 = ds_target.resample(date = '1MS').quantile(.05).rename("QObs(mm/d)_q05")

    ds_month = xr.merge([
        ds_target_month_mean,
        ds_target_month_max,
        ds_target_month_min,
        ds_target_month_q95.drop_vars('quantile'),
        ds_target_month_q05.drop_vars('quantile'),
        ds_forcings_month
    ])
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

    for attr in ['clim', 'topo', 'vege', 'soil', 'geol']:
        src = os.path.join(data_dir, '..', f'camels_{attr}.txt')
        dst = os.path.join('results', time_period, 'attributes', f'camels_{attr}.csv')
        df = pd.read_csv(src, sep = ';', dtype={0: str})  # make sure we read the basin id as str
        # df = df.fillna(df.mean())
        df.to_csv(dst, index = False)

# ##################################### #
# 3 - NH configuration                  #
# ##################################### #

representer.RoundTripRepresenter.ignore_aliases = lambda x, y: True

# These match the attributes listed in Appendix A: https://doi.org/10.5194/hess-23-5089-2019
climatic_attributes = ['p_mean', 'pet_mean', 'aridity', 'p_seasonality', 'frac_snow', 'high_prec_freq', 'high_prec_dur', 'low_prec_freq', 'low_prec_dur']
topographic_attributes = ['elev_mean', 'slope_mean', 'area_gages2']
landcover_attributes = ['frac_forest', 'lai_max', 'gvf_max', 'gvf_diff']
soil_attributes = ['soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity', 'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac']
hydrogeology_attributes = ['carbonate_rocks_frac', 'geol_permeability']

static_attributes = climatic_attributes + \
    topographic_attributes + \
    landcover_attributes + \
    soil_attributes + \
    hydrogeology_attributes

# To begin, use the Maurer forcings (as in https://doi.org/10.5194/hess-23-5089-2019)
dynamic_inputs = ['Dayl_s_maurer', 'PRCP_mm_day_maurer', 'SRAD_W_m2_maurer', 'Tmax_C_maurer', 'Tmin_C_maurer', 'Vp_Pa_maurer']
target_variables = ['QObs_mm_d_mean', 'QObs_mm_d_q95', 'QObs_mm_d_q05']

yaml = YAML() #typ = 'safe')
cfg = yaml.load(Path('/Users/simonmoulds/dev/neuralhydrology/examples/01-Introduction/1_basin.yml'))
cfg['experiment_name'] = 'test'
cfg['run_dir'] = None
cfg['train_basin_file'] = os.path.join(outputdir, "531_basin_list.txt") # TODO 531_basin_list.txt
cfg['validation_basin_file'] = os.path.join(outputdir, "531_basin_list.txt")
cfg['test_basin_file'] = os.path.join(outputdir, "531_basin_list.txt")
cfg['train_start_date'] = "01/10/1999"
cfg['train_end_date'] = "30/09/2008"
cfg['validation_start_date'] = "01/10/1980"
cfg['validation_end_date'] = "30/09/1989"
cfg['test_start_date'] = "01/10/1989"
cfg['test_end_date']= "30/09/1999"
cfg['device'] = "cpu" # "cuda:0"
cfg['validate_every'] = None # int(3)
cfg['validate_n_random_basins'] = int(1)
cfg['metrics'] = ['NSE']            # Might be different depending on daily/monthly?
cfg['save_validation_results'] = False
cfg['model'] = "cudalstm"
cfg['head'] = "regression"
cfg['output_activation'] = "linear" # Or relu, softplus
cfg['hidden_size'] = int(256)       # 128/64/... may vary by timescale
cfg['initial_forget_bias'] = int(3)
cfg['output_dropout'] = 0.4
cfg['optimizer'] = "Adam"
cfg['loss'] = "NSE"
cfg['learning_rate'] = {0: 1e-3, 10: 5e-4, 25: 1e-4}
cfg['batch_size'] = int(256)
cfg['epochs'] = int(30)
cfg['clip_gradient_norm'] = int(1)
cfg['predict_last_n'] = int(1)
cfg['seq_length'] = 365 # 12
cfg['num_workers'] = int(16)
cfg['log_interval'] = int(5)
cfg['log_tensorboard'] = True
cfg['log_n_figures'] = int(0)
cfg['save_weights_every'] = int(1)
cfg['dataset'] = "generic"
cfg['data_dir'] = outputdir
cfg['dynamic_inputs'] = dynamic_inputs
cfg['target_variables'] = target_variables
cfg['clip_targets_to_zero'] = target_variables
cfg['static_attributes'] = static_attributes

for time_period in ['day', 'month', 'season']:
    # The above configuration settings need to be adjusted for monthly-seasonal data
    if time_period == 'month':
        cfg["train_start_date"] = "01/10/1999"
        cfg["train_end_date"] = "01/09/2008"
        cfg["validation_start_date"] = "01/10/1980"
        cfg["validation_end_date"] = "01/09/1989"
        cfg["test_start_date"] = "01/10/1989"
        cfg["test_end_date"] = "01/09/1999"
        cfg["seq_length"] = 12
    if time_period == 'season':
        cfg["train_start_date"] = "01/12/1999" # DJF
        cfg["train_end_date"] = "01/09/2008"
        cfg["validation_start_date"] = "01/12/1980"
        cfg["validation_end_date"] = "01/09/1989"
        cfg["test_start_date"] = "01/12/1989"
        cfg["test_end_date"] = "01/09/1999"
        cfg["seq_length"] = 4
    cfg['run_dir'] = os.path.join('results', time_period, 'lstm', 'runs')
    cfg['train_basin_file'] = os.path.join('results', time_period, '531_basin_list.txt')
    cfg['test_basin_file'] = os.path.join('results', time_period, '531_basin_list.txt')
    cfg['validation_basin_file'] = os.path.join('results', time_period, '531_basin_list.txt')
    cfg['data_dir'] = os.path.join('results', time_period)

    conf_filename = os.path.join('results', time_period, 'basins.yml')
    with open(conf_filename, 'wb') as f:
        yaml.dump(cfg, f)

