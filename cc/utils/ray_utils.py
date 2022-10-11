import ray 

def if_is_actor(obj, method, blocking=True, *args, **kwargs):
    method = getattr(obj, method)
    if isinstance(obj, ray.actor.ActorHandle):
        obj_ref = getattr(method, "remote")(*args, **kwargs)
        if blocking:
            return ray.get(obj_ref)
        else:
            return obj_ref
    else:
        return method(*args, **kwargs)

def SyncOrAsyncClass(AsyncClass, SyncClass):
    if ray.is_initialized():
        return ray.remote(AsyncClass)
    else:
        return SyncClass

        