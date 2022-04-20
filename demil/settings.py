from os.path import dirname
import os
from pathlib import Path

# ===================== General ======================

ROOT = dirname(dirname(__file__))
encoding = "utf-8"
verbose = False

DATA_PATH = Path(ROOT, "data")
DEPRESSION_CORPUS = Path(DATA_PATH, "depression_corpus")
AVEC_AUDIO = Path(DATA_PATH, "avec2019", "audio_deep_features")
AVEC_VIDEO = Path(DATA_PATH, "avec2019", "video_CNNs")
AVEC_METADATA = Path(DATA_PATH, "avec2019", "metadata.csv")
PATH_TO_INSTAGRAM_DATA = Path(DATA_PATH, "instagram")

# ===================== LOSADA =====================

PATH_TO_ERISK = Path(DATA_PATH, "eRisk2021")
PATH_TO_ERISK = Path(DATA_PATH, "eRisk2021")
PATH_TO_LOSADA2016 = Path(DATA_PATH, "LOSADA2016")
PATH_TO_ERISKLOSADA = Path(DATA_PATH, "eRisk+LOSADA")

# ===================== LOSADA =====================

PATH_TO_DEPRESSBR = Path(DATA_PATH, "DepressBR")
MODELS_PATH = Path(ROOT, "models")
LOG_PATH = Path(ROOT, "data", "logs", "logs.txt")
LANGUAGE_MODEL = "neuralmind/bert-base-portuguese-cased"
VISUAL_MODEL = "resnet34"
WORKERS = 4
MAX_SEQ_LENGTH = 60
PATH_TO_BERT_CONFIG = Path(ROOT, "config", "config.json")

BERTIMBAU = [
    "neuralmind/bert-base-portuguese-cased",
    "neuralmind/bert-large-portuguese-cased",
]
ENGLISH_BERT = [
    "bert-base-cased",
    "bert-base-uncased"
]
XLM = ["xlm-roberta-base", "xlm-roberta-large"]


# ============= Create necessary folders =============

Path(ROOT, "models").mkdir(parents=True, exist_ok=True)  # Model checkpoints or features
Path(ROOT, "reports").mkdir(
    parents=True, exist_ok=True
)  # Documents with text and charts documenting the results
Path(ROOT, "scripts").mkdir(
    parents=True, exist_ok=True
)  # Scripts to be executed manually
Path(ROOT, "data").mkdir(parents=True, exist_ok=True)  # The data used in the experiments
Path(ROOT, "notebooks").mkdir(parents=True, exist_ok=True)
Path(LOG_PATH).parents[0].mkdir(parents=True, exist_ok=True)
DEPRESSION_CORPUS.mkdir(parents=True, exist_ok=True)