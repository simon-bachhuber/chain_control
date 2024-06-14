# Installation
1. Create a new conda environment for this project using `conda create -n chain_control python=3.10`
2. Git clone this repository
3. Go into the root folder in this repository and use `pip install -e .`
4. Install `fire` using `pip install fire`

# Usage

## Train controller
`python train_controller.py output_path train_sys train_refs`

where
- output_path: is the folder where the trained controller is stored
- train_sys: is the folder that contains all the dynamical systems used for training
- train_refs: is the folder that contains all the references used for training

Positional parameters can be given with e.g.
`python train_controller.py output_path train_sys train_refs --n-episodes 100`

## Evaluate trained controller 
`python eval_controller.py controller_output_path train_sys/dynamics1.npy val_refs/ eval_output_path`

where 
- controller_output_path: is the path to the controller to be evaluated
- train_sys/dynamics1.npy: is the path to the dynamics that we want to evaluate on
- val_refs: is the folder that contains the references that we want to track
- eval_output_path: is the folder where the observed input/output data is stored
