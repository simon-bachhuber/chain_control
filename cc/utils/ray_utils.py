import ray


def if_ray_actor(
    obj,
    method,
    *args,
    blocking=True,
    **kwargs,
):
    method = getattr(obj, method)
    if isinstance(obj, ray.actor.ActorHandle):
        obj_ref = getattr(method, "remote")(*args, **kwargs)
        if blocking:
            return ray.get(obj_ref)
        else:
            return obj_ref
    else:
        return method(*args, **kwargs)
