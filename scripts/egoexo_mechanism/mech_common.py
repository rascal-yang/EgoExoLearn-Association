import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/egoexo_textalign'))
from common import EXP as BASE,REP as BASE_REP,read,atomic,digest,CKPT,cool
EXP=ROOT/'experiments/egoexo_mechanism_20261025'
REP=ROOT/'reports/egoexo_mechanism_20261025'
ARMS=('N','V','M');SEEDS=(17,29,43)
