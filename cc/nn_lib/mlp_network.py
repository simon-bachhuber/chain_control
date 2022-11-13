import equinox as eqx
import jax.random as jrand


def mlp_network(
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
) -> eqx.Module:
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
