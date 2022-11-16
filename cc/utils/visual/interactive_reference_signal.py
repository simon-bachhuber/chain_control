from multiprocessing import Process, Value
from tkinter import Button, Scale, Tk, mainloop

import jax.numpy as jnp
import jax.tree_util as jtu

from cc import load
from cc.config import ROOT_DIR
from cc.env import make_env
from cc.utils.visual.viewer import launch_viewer_controller

from cc.env.collect.source import ObservationReferenceSource
from cc.env.wrappers import AddRefSignalRewardFnWrapper, TimelimitControltimestepWrapper

env = make_env("two_segments_v2", random=1, time_limit=10.0)  # <- CHANGE THIS LINE
controller = load(ROOT_DIR.joinpath("ctrb_v2.pkl"))  # <- CHANGE THIS LINE
JIT = False  # <- CHANGE THIS LINE


# SCRIPT BEGINS HERE

shared_reference_signal = Value("d", 0.0)


class InteractiveSource(ObservationReferenceSource):
    def __init__(self, ts: jnp.ndarray, yss, shared_value):
        super().__init__(ts, yss)
        self._shared_value = shared_value

    def get_reference_actor(self):
        return jtu.tree_map(
            lambda arr: arr * self._shared_value.value, super().get_reference_actor()
        )


def make_interactive_env(pipe_conn, env: TimelimitControltimestepWrapper):
    ts = env.ts
    N = len(ts) + 1
    ys_ones = jtu.tree_map(
        lambda arr: jnp.ones((1, N, *arr.shape)), env.observation_spec()
    )
    source = InteractiveSource(ts, ys_ones, pipe_conn)
    env = AddRefSignalRewardFnWrapper(env, source)
    return env


def launch_viewer(shared_reference_signal):
    env_w_shared_value = make_interactive_env(shared_reference_signal, env)
    launch_viewer_controller(env_w_shared_value, controller, jit=JIT)


p1 = Process(target=launch_viewer, args=[shared_reference_signal])
p1.start()


def tkinter_loop(shared_reference_signal):
    master = Tk()
    w1 = Scale(master, from_=-10.0, to=10.0)
    w1.pack()

    def send_over_pipe():
        shared_reference_signal.value = w1.get()

    Button(master, text="Send reference signal", command=send_over_pipe).pack()
    mainloop()


p2 = Process(target=tkinter_loop, args=[shared_reference_signal])
p2.start()

p1.join()
p2.join()
