import sys,json,hashlib,os,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/egoexo_textalign'))
from common import EXP as BASE,read,atomic,digest,cool
EXP=ROOT/'experiments/egoexo_states_20261025';REP=ROOT/'reports/egoexo_states_20261025'
MBASE=ROOT/'experiments/egoexo_mechanism_20261025';SEEDS=(17,29,43);ARMS=('temporal_mean','ordered_states');FRAMES=16

def direction(c):
 import re
 objects={'bowl':r'\bbowl\b|\bbasin\b','plate':r'\bplate\b|\bdish\b','spoon':r'\bspoon\b|\bladle\b','knife':r'\bknife\b','chopsticks':r'\bchopsticks\b','pot':r'\bpot\b|\bpan\b','lid':r'\blid\b','cup':r'\bcup\b'}
 c=c.lower();obj=[k for k,v in objects.items() if re.search(v,c)]
 if len(obj)!=1:return None
 action=c.replace('cutting board','board').replace('chopping board','board')
 if re.search(r'\band\b|\bthen\b|\bwhile\b|\bstir\w*\b|\bcut\w*\b|\bchop\w*\b|\bpour\w*\b',action):return None
 up=bool(re.search(r'\bpick\w* up\b|\btake\b|\blift\w*\b',action));down=bool(re.search(r'\bput\b.*\bdown\b|\bplace\w*\b|\bset\b.*\bdown\b',action))
 return (obj[0],int(down)) if up!=down else None
