import copy

import jax.numpy as jnp
import numpy as np
import pytest

from ..make_env import make_env
from .collect import sample_feedforward_collect_and_make_source
from .source import constant_after_transform_source


def dummy_env():
    return make_env("two_segments_v1", random=1)


def test_source():
    env = dummy_env()
    source = sample_feedforward_collect_and_make_source(env, seeds=[0, 1])[0]

    refs = source.get_references()
    flat_refs = refs["xpos_of_segment_end"]
    assert isinstance(flat_refs, np.ndarray)
    assert flat_refs.ndim == 3
    assert flat_refs.shape[0] == 2

    assert isinstance(
        source.get_references_for_optimisation()["xpos_of_segment_end"], jnp.ndarray
    )


def test_constant_after_transform_source():
    env = dummy_env()
    source = sample_feedforward_collect_and_make_source(env, seeds=[0, 1])[0]

    with pytest.raises(Exception):
        constant_after_transform_source(
            source, 3.0, new_time_limit=env.time_limit - env.control_timestep
        )

    with pytest.raises(Exception):
        temp = copy.deepcopy(source)
        temp._ts = None
        constant_after_transform_source(temp, 3.0)

    new_source = constant_after_transform_source(source, 3.0, new_time_limit=15.0)
    new_yss = new_source.get_references()["xpos_of_segment_end"]
    old_yss = source.get_references()["xpos_of_segment_end"]
    for i in range(2):
        assert np.all(new_yss[i, :300] == old_yss[i, :300])
        assert np.all(new_yss[i, 300] == new_yss[i, 300:])

    assert new_yss.shape[1] == 1501
    assert new_source._ts[-1] == 14.99
