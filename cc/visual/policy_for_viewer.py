from acme import core 
import copy 


def policy_for_viewer(actor: core.Actor):

    actor = copy.deepcopy(actor)

    # TODO super hacky
    if actor._adder:
        #print("WARNING: The `adder` has been GC. `adder` is not supported for `viewer`")
        actor._adder = None

    class _policy:
        observe_first=True
        last_action=None 

        def __call__(self, ts):

            if self.observe_first:
                actor.observe_first(ts)
                self.observe_first = False
            else:
                # observe now from last iteration
                actor.observe(self.last_action, next_timestep=ts)
            
            action = actor.select_action(ts.observation)
            self.last_action=action

            return action 

        def reset(self):
            self.observe_first=True

    return _policy()

