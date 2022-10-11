"""Functions to manage the common assets for domains."""

import os
from dm_control.utils import io as resources

_SUITE_DIR = os.path.dirname(os.path.dirname(__file__))
_FILENAMES = [
    "./common/materials.xml",
    "./common/skybox.xml",
    "./common/visual.xml",
]

ASSETS = {filename: resources.GetResource(os.path.join(_SUITE_DIR, filename))
          for filename in _FILENAMES}


def read_model(model_filename):
  """Reads a model XML file and returns its contents as a string."""
  return resources.GetResource(os.path.join(_SUITE_DIR, model_filename))