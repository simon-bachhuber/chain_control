<p align="center">
<img src="icons/icon2.png" height="200" />
</p>

# Chain Control (cc)

<img src="icons/coverage.svg" height="20" />

This package provides components to 
- simulate kinematic chains
- define and train a model of the dynamics
- define and train a controller that tracks a reference with the help of a model
- visualize the controller performance

A key concept is that models but also controllers are neural ODEs that are trained using gradient descent.

## Installation

Make sure that you first fulfill the following dependencies
- Requirements of `dm_control` under `Rendering` (https://github.com/deepmind/dm_control)
- `RecordVideoWrapper` requires `ffmpeg` to be available in terminal

Then, check out the setup script located under `scripts/first_setup.sh`. 

1. create a new conda environment for this project using `conda create -n chain_control python=3.10`
2. git clone this repository
3. cd into the `scripts` folder and use `source first_setup.sh` file

## Documentation

Check out the introductory notebooks located under `/docs` and the examples under `/cc/examples`

## Bugfixes

    Bug: "Mujoco-Lib could not be found"
    Solution: Re-install `dm_control`
    Steps:
        - pip uninstall dm_control
        - pip install dm_control
