import os


INHAND_HOME = os.environ.get("INHAND_HOME")
MODEL_PATH = os.path.join(INHAND_HOME, "models")
YML_PATH = os.path.join(MODEL_PATH, "yml")
SDF_PATH = os.path.join(MODEL_PATH, "sdf")
CONFIG_PATH = os.path.join(MODEL_PATH, "config")

DRAKE_HOME = os.environ.get("DRAKE_HOME")
