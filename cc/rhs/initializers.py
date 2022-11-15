from typing import Tuple

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrand

from ..core import PRNGKey


def state_init(key, state_dim):
    return jnp.zeros((state_dim,))


def eqx_linear_layer_init(key, in_features, out_features):
    import math

    lim = 1 / math.sqrt(in_features)
    weight = jrand.uniform(key, (out_features, in_features), minval=-lim, maxval=lim)
    return weight


def ABCD_init(key, state_size, input_size, output_size):
    key, c1, c2, c3 = jrand.split(key, 4)
    params = (
        eqx_linear_layer_init(c1, state_size, state_size),
        eqx_linear_layer_init(c2, input_size, state_size),
        eqx_linear_layer_init(c3, state_size, output_size),
        jnp.zeros((output_size, input_size)),
    )
    return params


def Network(
    in_size,
    out_size,
    width,
    depth,
    act_fn,
    act_fn_final,
    use_bias,
    use_dropout,
    dropout_rate,
    key,
):
    layers = []
    sizes = [in_size] + depth * [width] + [out_size]

    for i, (s_in, s_out) in enumerate(zip(sizes[:-1], sizes[1:])):

        if use_dropout:
            layers.append(eqx.nn.Dropout(dropout_rate))

        key, consume = jrand.split(key)
        layers.append(eqx.nn.Linear(s_in, s_out, use_bias, key=consume))

        if i < len(sizes) - 2:
            layers.append(eqx.nn.Lambda(act_fn))

    if use_dropout:
        layers.append(eqx.nn.Dropout(dropout_rate))

    layers.append(eqx.nn.Lambda(act_fn_final))

    return eqx.nn.Sequential(layers)


def f_g_init(key: PRNGKey, c) -> Tuple[eqx.Module, eqx.Module]:
    # TODO 
    key = jrand.PRNGKey(1,)
    c1, c2 = jrand.split(key, 2)
    print(c1)

    f = Network(
        c.state_size + c.input_size,
        c.state_size,
        c.width_f,
        c.depth_f,
        c.act_fn_f,
        c.act_final_f,
        c.use_bias_f,
        c.use_dropout_f,
        c.dropout_rate_f,
        key=c1,
    )
    g = Network(
        c.state_size,
        c.output_size,
        c.width_g,
        c.depth_g,
        c.act_fn_g,
        c.act_final_g,
        c.use_bias_g,
        c.use_dropout_g,
        c.dropout_rate_g,
        key=c2,
    )
    return f, g
