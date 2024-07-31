import fnmatch
import os

import setuptools


def find_data_files(package_dir, patterns, excludes=()):
    """Recursively finds files whose names match the given shell patterns."""
    paths = set()

    def is_excluded(s):
        for exclude in excludes:
            if fnmatch.fnmatch(s, exclude):
                return True
        return False

    for directory, _, filenames in os.walk(package_dir):
        if is_excluded(directory):
            continue
        for pattern in patterns:
            for filename in fnmatch.filter(filenames, pattern):
                # NB: paths must be relative to the package directory.
                relative_dirpath = os.path.relpath(directory, package_dir)
                full_path = os.path.join(relative_dirpath, filename)
                if not is_excluded(full_path):
                    paths.add(full_path)
    return list(paths)


setuptools.setup(
    name="chain_control",
    packages=setuptools.find_packages(),
    version="0.10.14",
    package_data={
        "cc": find_data_files(
            "cc", patterns=["*.xml", "*.m", "*.eqx"]
        )  # .m is not required right now
    },
    include_package_data=True,
    install_requires=[
        # after installation run `pip list`
        # if dm_control==0.4345734 version, then
        # `pip install --upgrade dm_control`
        # "dm_acme[jax]",
        # "dm_acme[envs]",
        "flax",
        "dm_control",
        "cloudpickle",
        "optax",
        "scikit-learn",
        "ray",
        "equinox==0.9.0",
        "tqdm",
        "ipykernel",
        "ipywidgets",  # otherwise tqdm will warn
        "imt-tree-utils",
        "myplotlib @ git+https://github.com/SimiPixel/myplotlib.git",
        "imageio[ffmpeg]",
        "control",
    ],
)
