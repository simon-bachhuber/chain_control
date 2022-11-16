import os
from pathlib import Path


DEFAULT_ROOT_DIR = "/home/simon/Documents/PYTHON/chain_control"
ROOT_DIR = Path(os.environ.get("PATH_CHAIN_CONTROL", DEFAULT_ROOT_DIR))
DEFAULT_RESULTS_DIR = "results"
RESULTS_DIR = ROOT_DIR.joinpath(DEFAULT_RESULTS_DIR)


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
