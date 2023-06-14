#!/usr/bin/env python3

import os
import torch
from neuralhydrology.nh_run import start_run, eval_run
from pathlib import Path
from ruamel.yaml import YAML

# TESTING
nh_config_filename = 'results/month/basins.yml'
# output_dir = 'results/month/lstm'

# # Get command line arguments
# nh_config = sys.argv[1]
# outputdir = sys.argv[2]

if torch.cuda.is_available():
    # Run on GPU
    start_run(config_file = Path(nh_config_filename))
else:
    # Run on CPU instead
    start_run(config_file = Path(nh_config_filename), gpu = -1)

# Neural Hydrology assigns a unique name to the output run
# directory. We find this name by identifying the most recent
# directory, then evaluate the model output.
yaml = YAML(typ = 'safe')
nh_config = yaml.load(Path(nh_config_filename))
base_run_dir = nh_config['run_dir']
experiment_name = nh_config['experiment_name']
run_dirs = [
    os.path.join(base_run_dir, d) for d in os.listdir(base_run_dir)
    if  os.path.isdir(os.path.join(base_run_dir, d)) & d.startswith(experiment_name)
]
run_dirs.sort(key = lambda x: os.path.getmtime(x))
run_dir = run_dirs[-1]

with open(os.path.join(base_run_dir, 'LATEST'), 'w') as f:
    f.write(run_dir)

