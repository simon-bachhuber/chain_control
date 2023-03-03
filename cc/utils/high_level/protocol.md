# Ackermann

```python

import jax.numpy as jnp
import jax
import optax 
from cc.utils.high_level.defaults import Env

envv = Env("ackermann", {}, {"MAX_STEER": np.deg2rad(30)}, 0.02)

make_masterplot(
    envv, f"ackermann_trained_noise_0_02_new_version2.pdf", False, experiment_id="trained", 
    model_kwargs={"state_dim": 25, "f_depth": 1, "f_width_size": 25, "f_final_activation": jax.nn.tanh}, 
    model_optimizer=envv.optimizer(0.001, 0.5, 0.05), model_l2=0.02, controller_n_steps=1000, 
    controller_kwargs={"state_dim": 25, "f_depth": 0, "f_width_size": 25},
    controller_optimizer=envv.optimizer(0.003, 0.75, 0.25), controller_noise_scale=0.02, controller_l2=0.0,
    dump_controller=True
)

{'training_rmse_model': DeviceArray(0.08511174, dtype=float32),
 'test_rmse_model': DeviceArray(0.1103202, dtype=float32),
 'train_rmse_controller': 0.519933,
 'test_rmse_controller': 0.63586247,
 'high_amplitude_rmse': 2.5662792,
 'double_steps_rmse': 0.99269265,
 'smooth_refs_rmse': 0.15473688,
 'smooth_to_constant_refs_rmse': 0.18177202}
```

# Pendulum 1

`test_rmse_controller = 0.53`
```python
{
    "model_state_dim": 75,
    "controller_state_dim": 50,
}
```

# Pendulum 2

`test_rmse_controller = 0.60`
```python
search_space = {
    "env_id": "pendulum2",
    "data_config_id": "2min",
    "model_state_dim": 75,
    "controller_state_dim": 15,
    "l2": 0.0,
    "controller_noise_scale": 0.0,
    "seed": 1,
    "controller_lr": 3e-3,
    "controller_clip": 0.05,
    "alpha": 1e-1,
}
```

# Pendulum 3

I choose gridsearch number `482`
