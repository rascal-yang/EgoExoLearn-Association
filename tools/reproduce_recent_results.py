"""Recompute headline arithmetic from summaries and paired timing samples."""
import json
import math
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'results/recent_20261028'
SEEDS=('17','29','43')


def read(name):return json.loads((ROOT/name).read_text())


def check(value,expected):
    assert math.isclose(value,expected,rel_tol=1e-10,abs_tol=1e-8),(value,expected)


def counted_table(filename,questions):
    archive=read(filename)
    print(f'\n{filename}: {questions} questions, epoch4, seeds17/29/43')
    for arm,model in archive['models'].items():
        values=[model[s]['correct']/questions*100 for s in SEEDS]
        for seed,value in zip(SEEDS,values):check(value,model[seed]['accuracy'])
        check(statistics.mean(values),model['mean_accuracy'])
        print(f'{arm}: {statistics.mean(values):.3f}%  '+', '.join(f'{v:.3f}' for v in values))


def main():
    counted_table('semantic_val.json',800)
    test=read('semantic_test.json')
    print(f'\nsemantic_test.json: {test["questions"]} questions')
    for arm,model in test['models'].items():
        values=[n/test['questions']*100 for n in model['per_seed_correct']]
        check(statistics.mean(values),model['mean_accuracy'])
        print(f'{arm}: {statistics.mean(values):.3f}%')
    counted_table('block10_val.json',800)
    layer=read('layer96.json');benchmark=read('layer96_benchmark_raw.json')
    assert layer['questions']==256 and len(layer['conditions'])==25 and benchmark['completed']
    print('\nlayer96: fixed256; RGB FP32; 3 seeds; not full800 VAL')
    dense=layer['conditions']['dense']['mean_accuracy']
    for name,row in layer['conditions'].items():
        check(statistics.mean(row['per_seed_accuracy']),row['mean_accuracy'])
        check(row['mean_accuracy']-dense,row['gain_pp'])
        if name=='dense':continue
        timing={}
        for batch,record in benchmark['batches'].items():
            ratios=[s['candidate_cuda_ms']/s['baseline_cuda_ms'] for s in record['samples'][name]]
            timing[batch]=(1-statistics.median(ratios))*100
            check(timing[batch],row['timing'][batch]['median_latency_reduction_percent'])
        print(f'{name}: {row["mean_accuracy"]:.3f}% ({row["gain_pp"]:+.3f}pp); '
              f'latency reduction B1={timing["1"]:+.2f}%, B8={timing["8"]:+.2f}%')
    print('\nPASS: headline counts/means and paired latency arithmetic agree; no model rerun.')


if __name__=='__main__':main()
