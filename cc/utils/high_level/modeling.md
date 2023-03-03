## Two Segments

### damping = 0.015, stiffness=1.5

```python
df.sort_values("_metric").head(30)[['_metric', 'config/clip',
'config/f_depth', 'config/f_final_activation', 'config/f_width_size',
'config/global_clip', 'config/lambda_l1', 'config/lambda_l2',
'config/lr', 'config/state_dim']]
## OUTPUT
       _metric  config/clip  config/f_depth              config/f_final_activation  config/f_width_size  config/global_clip  config/lambda_l1  config/lambda_l2  config/lr  config/state_dim
2230  0.909768         0.05               0  <function <lambda> at 0x14577ca7c5e0>                   25                0.05              0.00              0.00      0.001                50
2231  0.909768         0.50               0  <function <lambda> at 0x14577ca7c5e0>                   25                0.05              0.00              0.00      0.001                50
2362  0.911060         0.05               0  <function <lambda> at 0x14577ca7c5e0>                   25                0.50              0.00              0.02      0.001                50
2242  0.911770         0.05               0  <function <lambda> at 0x14577ca7c5e0>                   25                0.50              0.00              0.00      0.001                50
2243  0.915970         0.50               0  <function <lambda> at 0x14577ca7c5e0>                   25                0.50              0.00              0.00      0.001                50
2351  0.919145         0.50               0  <function <lambda> at 0x14577ca7c5e0>                   25                0.05              0.00              0.02      0.001                50
2350  0.919145         0.05               0  <function <lambda> at 0x14577ca7c5e0>                   25                0.05              0.00              0.02      0.001                50
2260  0.922663         0.05               0                         <PjitFunction>                   25                0.05              0.02              0.00      0.001                50
2261  0.922663         0.50               0                         <PjitFunction>                   25                0.05              0.02              0.00      0.001                50
2273  0.922747         0.50               0                         <PjitFunction>                   25                0.50              0.02              0.00      0.001                50
2363  0.923270         0.50               0  <function <lambda> at 0x14577ca7c5e0>                   25                0.50              0.00              0.02      0.001                50
2272  0.923798         0.05               0                         <PjitFunction>                   25                0.50              0.02              0.00      0.001                50
2393  0.924323         0.50               0                         <PjitFunction>                   25                0.50              0.02              0.02      0.001                50
2381  0.924405         0.50               0                         <PjitFunction>                   25                0.05              0.02              0.02      0.001                50
2380  0.924405         0.05               0                         <PjitFunction>                   25                0.05              0.02              0.02      0.001                50
2284  0.930548         0.05               0                         <PjitFunction>                   25                0.05              0.10              0.00      0.001                50
2285  0.930548         0.50               0                         <PjitFunction>                   25                0.05              0.10              0.00      0.001                50
2514  0.931195         0.50               0                         <PjitFunction>                   25                0.05              0.10              0.10      0.001                50
2496  0.931743         0.05               0                         <PjitFunction>                   25                0.05              0.02              0.10      0.001                50
2497  0.931743         0.50               0                         <PjitFunction>                   25                0.05              0.02              0.10      0.001                50
2296  0.932167         0.05               0                         <PjitFunction>                   25                0.50              0.10              0.00      0.001                50
2392  0.932376         0.05               0                         <PjitFunction>                   25                0.50              0.02              0.02      0.001                50
2297  0.933260         0.50               0                         <PjitFunction>                   25                0.50              0.10              0.00      0.001                50
2526  0.934140         0.50               0                         <PjitFunction>                   25                0.50              0.10              0.10      0.001                50
2506  0.934468         0.50               0                         <PjitFunction>                   25                0.50              0.02              0.10      0.001                50
2416  0.936025         0.50               0                         <PjitFunction>                   25                0.50              0.10              0.02      0.001                50
2525  0.938226         0.05               0                         <PjitFunction>                   25                0.50              0.10              0.10      0.001                50
2505  0.940068         0.05               0                         <PjitFunction>                   25                0.50              0.02              0.10      0.001                50
2415  0.940298         0.05               0                         <PjitFunction>                   25                0.50              0.10              0.02      0.001                50
2596  0.941720         0.05               0                         <PjitFunction>                   25                0.05              0.02              0.20      0.001                50

## Actually we can get
import jax.numpy as jnp
import jax
import optax 

def optimizer_rover(lr, clip, global_clip):
    optimizer = optax.chain(
        optax.clip_by_global_norm(global_clip),
        optax.clip(clip),
        optax.adam(lr),
    )
    return optimizer

env_id = "two_segments"
damping = 1.5*1e-2
stiffness=1.5
env = make_env(env_id, physics_kwargs={"damping": damping, "stiffness": stiffness})

make_masterplot(env_id, env, f"{env_id}_damp_{damping}_stiff_{stiffness}_trained.pdf", True, "trained", 
    {"state_dim": 75, "f_depth": 0, "f_width_size": 25, "u_transform": jnp.arctan}, 
    3, 700, {"state_dim": 40, "f_depth": 0}, 1, 1000, model_optimizer=optimizer_rover(0.01, 0.05, 0.05))

## OUTPUT
{'training_rmse_model': DeviceArray(0.66859454, dtype=float32),
 'test_rmse_model': DeviceArray(0.52103347, dtype=float32),
 'train_rmse_controller': 0.44505924,
 'test_rmse_controller': 0.6867308,
 'high_amplitude': 2.464438,
 'double_steps': 1.2279183,
 'smooth_refs': 1.9372492,
 'smooth_to_constant_refs': 1.4096903}

## OR

make_masterplot(env_id, env, f"{env_id}_damp_{damping}_stiff_{stiffness}_trained.pdf", True, "trained", 
    {"state_dim": 75, "f_depth": 0, "f_width_size": 25, "u_transform": jnp.arctan}, 
    3, 700, {"state_dim": 40, "f_depth": 0}, 1, 1000, model_optimizer=optimizer_rover(0.01, 0.05, 0.05))

{'training_rmse_model': DeviceArray(0.6174759, dtype=float32),
 'test_rmse_model': DeviceArray(0.511267, dtype=float32),
 'train_rmse_controller': 0.42773965,
 'test_rmse_controller': 0.6851262,
 'high_amplitude': 2.4503672,
 'double_steps': 1.2375759,
 'smooth_refs': 1.8456837,
 'smooth_to_constant_refs': 1.3312474}
```

## Rover

The run that created these runs was defined by 
```python
def train_eval_model(env, lambda_l1, lambda_l2, optimizer, **model_kwargs):
    train_gp = list(range(12))
    val_gp = list(range(12, 15))
    train_cos = list(range(12))
    val_cos = [2.5, 4.5, 6.5]

    train_sample = sample_feedforward_and_collect(env, train_gp, train_cos)
    val_sample = sample_feedforward_and_collect(env, val_gp, val_cos)

    model_trainer = make_model(
        env,
	    train_sample,
        val_sample,
        model_kwargs,
        1,
	    700,
	    lambda_l1,
        lambda_l2,
        optimizer,
    )

    return model_trainer.trackers[0].best_metric()


def YOUR_FUNCTION(config: dict[str, Any]) -> float:
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.pop("global_clip")),
        optax.clip(config.pop("clip")),
        optax.adam(config.pop("lr")),
    )
    env = make_env("rover")
    val_rmse = train_eval_model(
        env, config.pop("lambda_l1"), config.pop("lambda_l2"), optimizer, **config
    )
    return val_rmse
```

### Forward

With default values, that is `drive=0.66`

```python
# HEAD
       _metric  config/clip  config/f_depth              config/f_final_activation  config/f_width_size  config/global_clip  config/lambda_l1  config/lambda_l2  config/lr  config/state_dim
23    0.070317         0.50               2                         <PjitFunction>                   25                0.50               0.0              0.00      0.001                10
1325  0.072017         0.05               2                         <PjitFunction>                   25                0.50               0.0              0.02      0.001                25
1317  0.074142         0.50               1                         <PjitFunction>                   25                0.05               0.0              0.02      0.001                25
8     0.075417         0.05               1                         <PjitFunction>                   25                0.05               0.0              0.00      0.001                10
9     0.075417         0.50               1                         <PjitFunction>                   25                0.05               0.0              0.00      0.001                10
1326  0.075465         0.50               2                         <PjitFunction>                   25                0.50               0.0              0.02      0.001                25
22    0.077013         0.05               2                         <PjitFunction>                   25                0.50               0.0              0.00      0.001                10
2176  0.077313         0.50               2                         <PjitFunction>                   25                0.50               0.0              0.02      0.001                50
14    0.079019         0.05               1  <function <lambda> at 0x149b8091c940>                   25                0.50               0.0              0.00      0.001                10
1210  0.079411         0.05               1  <function <lambda> at 0x149b8091c940>                   25                0.50               0.0              0.00      0.001                25
1211  0.080318         0.50               1  <function <lambda> at 0x149b8091c940>                   25                0.50               0.0              0.00      0.001                25
1320  0.080647         0.50               1  <function <lambda> at 0x149b8091c940>                   25                0.50               0.0              0.02      0.001                25
1313  0.081416         0.05               1  <function <lambda> at 0x149b8091c940>                   25                0.05               0.0              0.02      0.001                25
1206  0.081477         0.05               2                         <PjitFunction>                   25                0.05               0.0              0.00      0.001                25
1207  0.081477         0.50               2                         <PjitFunction>                   25                0.05               0.0              0.00      0.001                25
1202  0.082964         0.50               2  <function <lambda> at 0x149b8091c940>                   25                0.05               0.0              0.00      0.001                25
1201  0.082964         0.05               2  <function <lambda> at 0x149b8091c940>                   25                0.05               0.0              0.00      0.001                25
1199  0.083279         0.05               1  <function <lambda> at 0x149b8091c940>                   25                0.05               0.0              0.00      0.001                25
1200  0.083279         0.50               1  <function <lambda> at 0x149b8091c940>                   25                0.05               0.0              0.00      0.001                25
1318  0.085070         0.05               2                         <PjitFunction>                   25                0.05               0.0              0.02      0.001                25
1319  0.085070         0.50               2                         <PjitFunction>                   25                0.05               0.0              0.02      0.001                25
16    0.085297         0.05               2  <function <lambda> at 0x149b8091c940>                   25                0.50               0.0              0.00      0.001                10
17    0.086331         0.50               2  <function <lambda> at 0x149b8091c940>                   25                0.50               0.0              0.00      0.001                10
1314  0.086447         0.50               2  <function <lambda> at 0x149b8091c940>                   25                0.05               0.0              0.02      0.001                25
11    0.087572         0.50               2                         <PjitFunction>                   25                0.05               0.0              0.00      0.001                10
10    0.087572         0.05               2                         <PjitFunction>                   25                0.05               0.0              0.00      0.001                10
15    0.087688         0.50               1  <function <lambda> at 0x149b8091c940>                   25                0.50               0.0              0.00      0.001                10
2102  0.087802         0.50               1  <function <lambda> at 0x149b8091c940>                   25                0.05               0.0              0.00      0.001                50
2101  0.087802         0.05               1  <function <lambda> at 0x149b8091c940>                   25                0.05               0.0              0.00      0.001                50
1216  0.088949         0.05               1                         <PjitFunction>                   25                0.50               0.0              0.00      0.001                25

# TAIL
       _metric  config/clip  config/f_depth              config/f_final_activation  config/f_width_size  config/global_clip  config/lambda_l1  config/lambda_l2  config/lr  config/state_dim
2401  1.330226         0.05               2                         <PjitFunction>                   25                0.05               0.1              0.00     0.0001                50
2402  1.330226         0.50               2                         <PjitFunction>                   25                0.05               0.1              0.00     0.0001                50
2409  1.330226         0.50               2                         <PjitFunction>                   25                0.50               0.1              0.00     0.0001                50
1910  1.330244         0.05               1                         <PjitFunction>                   25                0.50               0.5              0.10     0.0001                25
2486  1.330262         0.05               2  <function <lambda> at 0x149b8091c940>                   25                0.05               0.1              0.02     0.0001                50
2398  1.330390         0.50               2  <function <lambda> at 0x149b8091c940>                   25                0.05               0.1              0.00     0.0001                50
1191  1.330724         0.05               0                         <PjitFunction>                   25                0.50               0.5              0.50     0.0001                10
1959  1.330899         0.50               1  <function <lambda> at 0x149b8091c940>                   25                0.05               0.5              0.20     0.0001                25
1971  1.330900         0.50               1  <function <lambda> at 0x149b8091c940>                   25                0.50               0.5              0.20     0.0001                25
1194  1.330907         0.50               1                         <PjitFunction>                   25                0.50               0.5              0.50     0.0001                10
1183  1.330908         0.05               1                         <PjitFunction>                   25                0.05               0.5              0.50     0.0001                10
1184  1.330908         0.50               1                         <PjitFunction>                   25                0.05               0.5              0.50     0.0001                10
1188  1.331048         0.50               1  <function <lambda> at 0x149b8091c940>                   25                0.50               0.5              0.50     0.0001                10
1178  1.331049         0.50               1  <function <lambda> at 0x149b8091c940>                   25                0.05               0.5              0.50     0.0001                10
1177  1.331049         0.05               1  <function <lambda> at 0x149b8091c940>                   25                0.05               0.5              0.50     0.0001                10
1187  1.331061         0.05               1  <function <lambda> at 0x149b8091c940>                   25                0.50               0.5              0.50     0.0001                10
1970  1.331186         0.05               1  <function <lambda> at 0x149b8091c940>                   25                0.50               0.5              0.20     0.0001                25
1193  1.331489         0.05               1                         <PjitFunction>                   25                0.50               0.5              0.50     0.0001                10
1965  1.331847         0.50               1                         <PjitFunction>                   25                0.05               0.5              0.20     0.0001                25
1964  1.331847         0.05               1                         <PjitFunction>                   25                0.05               0.5              0.20     0.0001                25
1977  1.331848         0.50               1                         <PjitFunction>                   25                0.50               0.5              0.20     0.0001                25
1976  1.332031         0.05               1                         <PjitFunction>                   25                0.50               0.5              0.20     0.0001                25
2090  1.337640         0.50               1  <function <lambda> at 0x149b8091c940>                   25                0.50               0.5              0.50     0.0001                25
2079  1.337640         0.50               1  <function <lambda> at 0x149b8091c940>                   25                0.05               0.5              0.50     0.0001                25
2078  1.337640         0.05               1  <function <lambda> at 0x149b8091c940>                   25                0.05               0.5              0.50     0.0001                25
2089  1.338019         0.05               1  <function <lambda> at 0x149b8091c940>                   25                0.50               0.5              0.50     0.0001                25
2084  1.339050         0.50               1                         <PjitFunction>                   25                0.05               0.5              0.50     0.0001                25
2083  1.339050         0.05               1                         <PjitFunction>                   25                0.05               0.5              0.50     0.0001                25
2096  1.339051         0.50               1                         <PjitFunction>                   25                0.50               0.5              0.50     0.0001                25
2095  1.339316         0.05               1                         <PjitFunction>                   25                0.50               0.5              0.50     0.0001                25
```

### Backwards
`drive=-0.55`

Here `lambda` is no activation, `Pjit` is `Tanh`
```python
df.sort_values("_metric").head(30)[['_metric', 'config/clip',
...        'config/f_depth', 'config/f_final_activation', 'config/f_width_size',
...        'config/global_clip', 'config/lambda_l1', 'config/lambda_l2',
...        'config/lr', 'config/state_dim']]
# -> output
       _metric  config/clip  config/f_depth              config/f_final_activation  config/f_width_size  config/global_clip  config/lambda_l1  config/lambda_l2  config/lr  config/state_dim
2426  0.304337         0.05               1  <function <lambda> at 0x152f5ae42d30>                   25                0.50               0.0              0.02      0.001                50
1332  0.313851         0.05               2  <function <lambda> at 0x152f5ae42d30>                   25                0.50               0.0              0.02      0.001                25
2425  0.315312         0.50               1                         <PjitFunction>                   25                0.05               0.0              0.02      0.001                50
1320  0.316226         0.05               2  <function <lambda> at 0x152f5ae42d30>                   25                0.05               0.0              0.02      0.001                25
1321  0.316226         0.50               2  <function <lambda> at 0x152f5ae42d30>                   25                0.05               0.0              0.02      0.001                25
2320  0.317942         0.05               2  <function <lambda> at 0x152f5ae42d30>                   25                0.50               0.0              0.00      0.001                50
2427  0.318307         0.50               1  <function <lambda> at 0x152f5ae42d30>                   25                0.50               0.0              0.02      0.001                50
2421  0.318919         0.50               1  <function <lambda> at 0x152f5ae42d30>                   25                0.05               0.0              0.02      0.001                50
1333  0.322392         0.50               2  <function <lambda> at 0x152f5ae42d30>                   25                0.50               0.0              0.02      0.001                25
2325  0.322470         0.50               1                         <PjitFunction>                   25                0.50               0.0              0.00      0.001                50
2319  0.323311         0.50               1  <function <lambda> at 0x152f5ae42d30>                   25                0.50               0.0              0.00      0.001                50
2318  0.324353         0.05               1  <function <lambda> at 0x152f5ae42d30>                   25                0.50               0.0              0.00      0.001                50
1339  0.325143         0.50               2                         <PjitFunction>                   25                0.50               0.0              0.02      0.001                25
2308  0.328860         0.50               1  <function <lambda> at 0x152f5ae42d30>                   25                0.05               0.0              0.00      0.001                50
2307  0.328860         0.05               1  <function <lambda> at 0x152f5ae42d30>                   25                0.05               0.0              0.00      0.001                50
2326  0.330160         0.05               2                         <PjitFunction>                   25                0.50               0.0              0.00      0.001                50
2324  0.334589         0.05               1                         <PjitFunction>                   25                0.50               0.0              0.00      0.001                50
1220  0.334605         0.05               2                         <PjitFunction>                   25                0.50               0.0              0.00      0.001                25
2428  0.335264         0.05               2  <function <lambda> at 0x152f5ae42d30>                   25                0.50               0.0              0.02      0.001                50
1327  0.335890         0.50               2                         <PjitFunction>                   25                0.05               0.0              0.02      0.001                25
1326  0.335890         0.05               2                         <PjitFunction>                   25                0.05               0.0              0.02      0.001                25
1338  0.336050         0.05               2                         <PjitFunction>                   25                0.50               0.0              0.02      0.001                25
2312  0.339777         0.05               1                         <PjitFunction>                   25                0.05               0.0              0.00      0.001                50
2313  0.339777         0.50               1                         <PjitFunction>                   25                0.05               0.0              0.00      0.001                50
2528  0.340425         0.50               1  <function <lambda> at 0x152f5ae42d30>                   25                0.50               0.0              0.10      0.001                50
2322  0.341011         0.05               0                         <PjitFunction>                   25                0.50               0.0              0.00      0.001                50
2516  0.341365         0.05               1  <function <lambda> at 0x152f5ae42d30>                   25                0.05               0.0              0.10      0.001                50
2517  0.341365         0.50               1  <function <lambda> at 0x152f5ae42d30>                   25                0.05               0.0              0.10      0.001                50
2534  0.341629         0.50               1                         <PjitFunction>                   25                0.50               0.0              0.10      0.001                50
2527  0.342043         0.05               1  <function <lambda> at 0x152f5ae42d30>                   25                0.50               0.0              0.10      0.001                50
############################################################
df.sort_values("_metric").tail(30)[['_metric', 'config/clip',
'config/f_depth', 'config/f_final_activation', 'config/f_width_size',
'config/global_clip', 'config/lambda_l1', 'config/lambda_l2',
'config/lr', 'config/state_dim']]
# -> output
       _metric  config/clip  config/f_depth              config/f_final_activation  config/f_width_size  config/global_clip  config/lambda_l1  config/lambda_l2  config/lr  config/state_dim
1427  0.569139         0.50               1  <function <lambda> at 0x152f5ae42d30>                   25                0.50               0.5              0.02     0.0010                25
579   0.569345         0.50               1  <function <lambda> at 0x152f5ae42d30>                   25                0.05               0.5              0.50     0.0010                10
578   0.569345         0.05               1  <function <lambda> at 0x152f5ae42d30>                   25                0.05               0.5              0.50     0.0010                10
591   0.569346         0.50               1  <function <lambda> at 0x152f5ae42d30>                   25                0.50               0.5              0.50     0.0010                10
1651  0.569365         0.05               1                         <PjitFunction>                   25                0.50               0.5              0.20     0.0010                25
1523  0.569422         0.50               1  <function <lambda> at 0x152f5ae42d30>                   25                0.05               0.5              0.10     0.0010                25
1532  0.569487         0.05               1  <function <lambda> at 0x152f5ae42d30>                   25                0.50               0.5              0.10     0.0010                25
1639  0.569536         0.05               1                         <PjitFunction>                   25                0.05               0.5              0.20     0.0010                25
1640  0.569536         0.50               1                         <PjitFunction>                   25                0.05               0.5              0.20     0.0010                25
1652  0.569536         0.50               1                         <PjitFunction>                   25                0.50               0.5              0.20     0.0010                25
1646  0.569913         0.50               1  <function <lambda> at 0x152f5ae42d30>                   25                0.50               0.5              0.20     0.0010                25
1633  0.569915         0.05               1  <function <lambda> at 0x152f5ae42d30>                   25                0.05               0.5              0.20     0.0010                25
1634  0.569915         0.50               1  <function <lambda> at 0x152f5ae42d30>                   25                0.05               0.5              0.20     0.0010                25
1645  0.569959         0.05               1  <function <lambda> at 0x152f5ae42d30>                   25                0.50               0.5              0.20     0.0010                25
2297  0.570264         0.50               1  <function <lambda> at 0x152f5ae42d30>                   25                0.50               0.5              0.50     0.0001                25
2284  0.570268         0.05               1  <function <lambda> at 0x152f5ae42d30>                   25                0.05               0.5              0.50     0.0001                25
2285  0.570268         0.50               1  <function <lambda> at 0x152f5ae42d30>                   25                0.05               0.5              0.50     0.0001                25
2303  0.570274         0.50               1                         <PjitFunction>                   25                0.50               0.5              0.50     0.0001                25
2291  0.570284         0.50               1                         <PjitFunction>                   25                0.05               0.5              0.50     0.0001                25
2290  0.570284         0.05               1                         <PjitFunction>                   25                0.05               0.5              0.50     0.0001                25
1771  0.570737         0.05               1                         <PjitFunction>                   25                0.50               0.5              0.50     0.0010                25
2302  0.570773         0.05               1                         <PjitFunction>                   25                0.50               0.5              0.50     0.0001                25
2296  0.570906         0.05               1  <function <lambda> at 0x152f5ae42d30>                   25                0.50               0.5              0.50     0.0001                25
1759  0.570984         0.05               1                         <PjitFunction>                   25                0.05               0.5              0.50     0.0010                25
1760  0.570984         0.50               1                         <PjitFunction>                   25                0.05               0.5              0.50     0.0010                25
1772  0.571015         0.50               1                         <PjitFunction>                   25                0.50               0.5              0.50     0.0010                25
1765  0.571263         0.05               1  <function <lambda> at 0x152f5ae42d30>                   25                0.50               0.5              0.50     0.0010                25
1753  0.571453         0.05               1  <function <lambda> at 0x152f5ae42d30>                   25                0.05               0.5              0.50     0.0010                25
1754  0.571453         0.50               1  <function <lambda> at 0x152f5ae42d30>                   25                0.05               0.5              0.50     0.0010                25
1766  0.571457         0.50               1  <function <lambda> at 0x152f5ae42d30>                   25                0.50               0.5              0.50     0.0010                25
```