#!/bin/bash

# this file should be execute from the root directory of this project
# using `./scripts/run_notebooks.sh`

cd docs
nbtb run --inplace 1_defining_an_environment.ipynb 
nbtb run --inplace 2_collecting_data_from_an_environment.ipynb 
nbtb run --inplace 3_training_a_model.ipynb 
nbtb run --inplace 4_training_a_model_and_controller.ipynb 
nbtb run --inplace 5_save_loading_model_controller.ipynb 

rm train_sample.pkl model.eqx controller.eqx replay_sample.pkl