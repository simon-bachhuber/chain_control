import pytest

from ...make_env import make_env
from .attribute import AttributeWrapper


def test_attribute_wrapper():
    env = make_env("two_segments_v1", random=1)

    with pytest.raises(Exception):
        env_wrapped = AttributeWrapper(env, action_spec=2.0)

    env_wrapped = AttributeWrapper(env, safe=False, action_spec=2.0)
    assert env_wrapped.action_spec == 2.0
