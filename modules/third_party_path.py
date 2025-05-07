import os
import sys

from modules.utils.constants import ROOT_DIR
def REPO_DIR(name):
    return os.path.abspath(os.path.join(ROOT_DIR, "modules", "third_party", name))

# REPO_DIR = lambda name: os.path.abspath(os.path.join(os.path.dirname(__file__), name))

paths = [
    REPO_DIR("Matcha-TTS"),
]


def setup_third_party_paths():
    for pth in paths:
        if pth not in sys.path:
            sys.path.insert(0, pth)
