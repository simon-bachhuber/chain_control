# Chain Control (cc)

<center>
<img src="icon2.png" height="200" />
</center>

This package provides components to 
- simulate kinematic chains
- define and train a model of the dynamics
- define and train a controller that tracks a reference with the help of a model
- visualize the controller performance

A key concept is that models but also controllers are neural ODEs that are trained using gradient descent.

## Installation

- create & activate new conda environment with Python 3.9
- git clone this repository
- in root of this repository: pip install -e . 

## Documentation

Check out the five introductory notebooks located under /docs

## Contributing / Pull requests

For information on how to create a pull request refer to
https://github.com/MarcDiethelm/contributing

## Modules

- cc.env.
    - envs 
        
        Contains the different simulation setups. Each setup consists of two files. `setup1.py` and `setup2.xml`
    - wrappers

        Contains many useful wrappers that modifies a simulation setup (or an Environment).

    - ...

## Bugfixes

    Bug: "Mujoco-Lib could not be found"
    Solution: Re-install `dm_control`
    Steps:
        - pip uninstall dm_control
        - pip install dm_control
