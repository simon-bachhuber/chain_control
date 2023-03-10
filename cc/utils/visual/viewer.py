import copy

import dm_env
from acme import core
from dm_control.viewer import launch

from ...env.collect import ModuleActor


def launch_viewer_controller(env, controller, jit=True, reset_every: int = -1):
    actor = ModuleActor(env.action_spec(), controller=controller, jit=jit)
    return launch_viewer(env, actor, reset_every)


def launch_viewer(env: dm_env.Environment, actor: core.Actor, reset_every: int = -1):
    return launch(env, _policy_for_viewer(actor, reset_every))


def _policy_for_viewer(actor: core.Actor, reset_every: int):
    actor = copy.deepcopy(actor)
    if actor._adder:
        actor._adder = None

    class _policy:
        observe_first = True
        last_action = None
        n_call_calls = 0

        def __call__(self, ts):
            if self.observe_first:
                actor.observe_first(ts)
                self.observe_first = False
            else:
                # observe now from last iteration
                actor.observe(self.last_action, next_timestep=ts)

            action = actor.select_action(ts.observation)
            self.last_action = action

            self.n_call_calls += 1
            if reset_every > 0:
                if (self.n_call_calls % reset_every) == 0:
                    self.reset()

            return action

        def reset(self):
            # this flag triggers an actor reset at next __call__
            self.observe_first = True

    return _policy()
