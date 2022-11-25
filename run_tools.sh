black cc
isort cc 
flake8
pytype --config pytype.cfg 
pytest

nbtb run --inplace docs/1_defining_an_environment.ipynb 
nbtb run --inplace docs/2_collecting_data_from_an_environment.ipynb 
nbtb run --inplace docs/3_training_a_model.ipynb 
nbtb run --inplace docs/4_training_a_model_and_controller.ipynb 
nbtb run --inplace docs/5_save_loading_model_controller.ipynb 

rm train_sample.pkl model.eqx controller.eqx replay_sample.pkl