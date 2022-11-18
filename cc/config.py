_use_tqdm = True


def use_tqdm() -> bool:
    return _use_tqdm


def enable_tqdm():
    global _use_tqdm
    _use_tqdm = True


def disable_tqdm():
    global _use_tqdm
    _use_tqdm = False


_print_compile_warn = True


def disable_compile_warn():
    global _print_compile_warn
    _print_compile_warn = False


def print_compile_warn():
    return _print_compile_warn


def force_cpu_backend():
    from jax import config
    config.update("jax_platform_name", "cpu")
    config.update("jax_platforms", "cpu")
