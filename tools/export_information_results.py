"""Export reviewed aggregates from an immutable information-mechanism run.

This does not rerun inference, relabel regions, select checkpoints or replace scores.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main(a):
    run=Path(a.run_dir);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    record=json.loads((run/'run.json').read_text());main=json.loads((run/'main_results.json').read_text())
    audit=json.loads((run/'audit/audit.json').read_text());region=json.loads((run/'region_effects.json').read_text())
    assert record['status']=='complete' and record['exit_code']==0 and main['lineage_passed'] and audit['passed']
    results=json.loads((run/'region_results.json').read_text())['rows'];by={}
    for r in results:by.setdefault((r['record_id'],r['uid'],r['role'],r['model']),{})[r['condition']]=r
    matched=[v for v in by.values() if v['clean']['cohort']=='random_query' and v['clean']['matched_analysis_eligible']]
    flips=[]
    for v in matched:
        for sigma in (4,8):
            target=v[f'roi_sigma{sigma}'];bg=[r for key,r in v.items() if key.startswith('bg') and key.endswith(f'sigma{sigma}')]
            if v['clean']['correct'] and not target['correct']:
                flips.append(dict(uid=v['clean']['uid'],model=v['clean']['model'],sigma=sigma,background_correct=sum(r['correct'] for r in bg),background_n=len(bg)))
    sign_counts={}
    for model in ('P_base','P_rich'):
        for sigma in (4,8):
            rows=[r for r in region['rows'] if r['cohort']=='random_query' and r['matched_analysis_eligible'] and r['model']==model and r['sigma']==sigma]
            sign_counts[f'{model}__sigma{sigma}']=dict(n=len(rows),positive_D=sum(r['D']>0 for r in rows),median_D=statistics.median(r['D'] for r in rows))
    paired={}
    for sigma in (4,8):
        v=[r['D_rich_minus_base'] for r in region['paired_supervision_effects'] if r['cohort']=='random_query' and r['matched_analysis_eligible'] and r['sigma']==sigma]
        paired[str(sigma)]=dict(n=len(v),mean=statistics.mean(v),positive=sum(x>0 for x in v),median=statistics.median(v))
    per_question=[json.loads(x) for x in (run/'per_question.jsonl').read_text().splitlines()]
    decomposition={}
    for model in ('P_base','P_rich'):
        d={}
        for condition in ('repeat4','nested16'):
            v=[r['conditions'][model+'__'+condition] for r in per_question]
            d[condition]=dict(positive=statistics.mean(r['positive'] for r in v),strongest_negative=statistics.mean(r['strongest_negative'] for r in v))
        d['change']=dict(positive=d['nested16']['positive']-d['repeat4']['positive'],strongest_negative=d['nested16']['strongest_negative']-d['repeat4']['strongest_negative'])
        decomposition[model]=d
    known=[r for r in region['rows'] if r['cohort']=='known_case']
    energy=json.loads((run/'audit/perturbation_energy.json').read_text());energy_summary={}
    eligible_ids={v['clean']['record_id'] for v in matched}
    known_ids={r['record_id'] for r in known}
    for sigma in (4,8):
        ratios=[]
        for rid in sorted(eligible_ids|known_ids):
            rows=[r for r in energy if r['record_id']==rid and r['sigma']==sigma]
            roi=next(r['changed_region_rgb_rms'] for r in rows if r['region']=='roi')
            bg=statistics.mean(r['changed_region_rgb_rms'] for r in rows if r['region'].startswith('bg'))
            ratios.append(dict(record_id=rid,roi_rms=roi,mean_background_rms=bg,ratio=roi/bg))
        energy_summary[str(sigma)]=ratios
    sources=['main_results.json','region_effects.json','region_results.json','per_question.jsonl','run.json',
             'lineage_check.json','audit/audit.json','audit/perturbation_energy.json']
    summary=dict(schema=1,run_id=record['run_id'],inference_commit=record['git_commit'],audit_commit=audit['audit_commit'],
                 export_script_sha256=sha(__file__),training_seed=17,training=False,test_evaluation=False,
                 source_hashes={p:sha(run/p) for p in sources},
                 summary=main['summary'],contrasts=main['contrasts'],conversion_counts=audit['conversions'],score_decomposition=decomposition,
                 regional_summary=region['summary'],regional_sign_counts=sign_counts,regional_paired_selectivity=paired,
                 matched_random_roi_correct_to_wrong=flips,known_cases=known,region_coverage=audit['region_coverage'],regional_input_energy=energy_summary,
                 extra_observations=audit['extra_observations'],numerical_checks=dict(lineage=json.loads((run/'lineage_check.json').read_text()),
                 score_max_abs=max(r['max_abs_score_error'] for r in audit['score_checks']),regional_clean_margin_max_abs=audit['regional_clean_margin_max_abs']),
                 limitations=['Single training seed; repeatedly used development VAL','16 input positions in both repeat4 and nested16; not a native 4-frame efficiency comparison',
                              'Assistant-defined regions; only 8/24 matched random queries, 7 ego and 1 exo','Known old-M cases are already incorrect for the P models',
                              'No positive supervision-by-observation interaction established; margin and accuracy interactions differ',
                              'Equal area and blur strength do not guarantee equal removed RGB variation','No unique semantic causal mechanism, temporal alignment or independent generalization proven'])
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(output=str(out/'summary.json'),known_case_clean_margins=[r['clean_margin'] for r in known[::2]],matched_random_roi_correct_to_wrong=len(flips)),ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--output-dir',required=True);main(p.parse_args())
