from ..types import *
import jax.random as jrand 


def state_init(key, state_dim):
    return jnp.zeros((state_dim,))


def eqx_linear_layer_init(key, in_features, out_features):
    import math 
    lim = 1 / math.sqrt(in_features)
    weight = jrand.uniform(
        key, (out_features, in_features), minval=-lim, maxval=lim
    )
    return weight 


def ABCD_init(key, state_size, input_size, output_size):
    key, c1, c2, c3 = jrand.split(key, 4)
    params = (
        eqx_linear_layer_init(c1, state_size, state_size),
        eqx_linear_layer_init(c2, input_size, state_size),
        eqx_linear_layer_init(c3, state_size, output_size),
        jnp.zeros((output_size, input_size))
    )
    return params 


def f_g_init(key: PRNGKey, c) -> Tuple[eqx.Module, eqx.Module]:
    key, c1, c2 = jrand.split(key, 3)

    # TODO 
    # c.use_bias is functionless right now 
    
    f = eqx.nn.MLP(
        c.state_size + c.input_size, c.state_size, c.width_f, c.depth_f, 
        c.act_fn_f, c.act_final_f, key=c1
    )
    g = eqx.nn.MLP(
        c.state_size, c.output_size, c.width_g, c.depth_g, 
        c.act_fn_g, c.act_final_g, key=c2
    )
    return f,g 

