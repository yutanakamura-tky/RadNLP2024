"""Shared paths and the archived offline metric, with no network dependency."""
import os,sys,zipfile
from pathlib import Path
import pandas as pd
PACKAGE=Path(__file__).resolve().parents[1]
DATA=Path(os.environ['RADNLP_DATA_DIR']).resolve()
OUTPUT=Path(os.environ['RADNLP_OUTPUT_DIR']).resolve()
sys.path.insert(0,str(PACKAGE/'vendor'))
from official_metric import MainTaskMetric,SubTaskMetric
NAMES={'en':'english','ja':'japanese'}
def gold(task,language,split='test'):
    with zipfile.ZipFile(DATA/'sources/RadNLP2024/dataset_for_hugging_face_hub.zip') as z:
        return pd.read_csv(z.open(f'dataset_for_hugging_face_hub/{task}/{NAMES[language]}_{split}.csv'))
def case_map():
    return pd.read_csv(DATA/'sources/RadNLP2024/all_corpus.csv').set_index('id').case_id
