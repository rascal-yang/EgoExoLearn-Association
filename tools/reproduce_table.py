"""Recompute the published comparison from seed-level counts; no GPU needed."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    four = json.loads((ROOT/'results/four_frames.json').read_text())['metrics']
    sixteen = json.loads((ROOT/'results/sixteen_frames.json').read_text())['metrics']
    print('| Method | 4 frames | 16 frames | Change (pp) |')
    print('|---|---:|---:|---:|')
    for method, key in [('M','M16'),('temporal_mean','temporal_mean'),('ordered_states','ordered_states')]:
        a = sum(four[method]['correct_counts'])/24
        b = sum(sixteen[key]['normal']['counts'])/24
        assert abs(a-four[method]['mean_accuracy'])<1e-9
        assert abs(b-sixteen[key]['normal']['mean_accuracy'])<1e-9
        print(f'| {method} | {a:.3f}% | {b:.3f}% | +{b-a:.3f} |')

if __name__ == '__main__':
    main()
