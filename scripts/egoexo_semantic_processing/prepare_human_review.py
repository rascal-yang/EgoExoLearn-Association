"""Export blinded review materials; all human labels remain explicitly empty."""
import csv
from PIL import Image
from posthoc_core import *


def main():
    target=REP/'人工核验材料';target.mkdir(exist_ok=True)
    if (target/'materials_complete.json').exists():return
    records=read(BASE/'records.json');phrases=read(BASE/'phrases.json');split=np.load(BASE/'splits.npz')
    locked=[int(i) for i in split['holdout'] if int(hashlib.sha256(records[int(i)]['key'][0].encode()).hexdigest()[:8],16)%3==0]
    train_frequency=collections.Counter(p for i in split['train'] for p in records[int(i)]['phrases'])
    def frequency(i):
        return min([train_frequency[p] for p in records[i]['phrases']] or [0])
    median=float(np.median([frequency(i) for i in locked]))
    pools=collections.defaultdict(list)
    for i in locked:
        v,a,b=records[i]['key'];duration=0 if b-a<=5 else (1 if b-a<=15 else 2)
        pools[(records[i]['view'],duration,'lower' if frequency(i)<=median else 'upper')].append(i)
    total=min(200,len(locked));strata=sorted(pools)
    raw=np.array([len(pools[s])*total/len(locked) for s in strata]);quotas=np.floor(raw).astype(int)
    for j in np.argsort(-(raw-quotas),kind='stable')[:total-int(quotas.sum())]:quotas[j]+=1
    rng=np.random.default_rng(20261027);chosen=[]
    for s,n in zip(strata,quotas):chosen.extend(map(int,rng.choice(pools[s],int(n),replace=False)))
    assert len(chosen)==total and len(set(chosen))==total
    plan=dict(clips=chosen,source='Locked review partition from the original internal holdout, video-hash mod3 ==0',
              selection='score-independent proportional stratification by view, duration and TRAIN phrase frequency',
              strata=[dict(stratum=list(s),pool=len(pools[s]),quota=int(n)) for s,n in zip(strata,quotas)],
              regions_two_independent_raters=True,labels_filled=False,prior_M_holdout_selection_exposure=True,
              sources={str(p.relative_to(ROOT)):digest(p) for p in [BASE/'records.json',BASE/'splits.npz',REL/'review_plan.json',Path(__file__)]})
    atomic(target/'material_plan.json',plan)
    images=target/'frames';images.mkdir(exist_ok=True)
    import backend as b
    b.ref.init_worker('egoexo')
    decoded=b.ref.bounded_map(decode16,[records[i]['key'] for i in chosen],workers=4,initializer=b.ref.init_worker,initargs=('egoexo',))
    region_rows=[]
    for j,i in enumerate(chosen):
        if any((EXP/x).exists() for x in ('STOP','THERMAL_STOP.json')):raise StudyStop('Human-material export stopped')
        image,pts=next(decoded);first=np.unique(pts,return_index=True)[1];first.sort()
        times=first[np.rint(np.linspace(0,len(first)-1,min(4,len(first)))).astype(int)]
        record=records[i]
        for frame in times:
            rgb=image[:,frame]*np.array([68.5005327,66.6321579,70.32316305])[:,None,None]+np.array([122.7709393,116.7460125,104.09373615])[:,None,None]
            filename=f'clip{j:03d}_f{int(frame):02d}.png'
            Image.fromarray(np.clip(np.rint(rgb.transpose(1,2,0)),0,255).astype(np.uint8)).save(images/filename)
            nouns='; '.join(phrases[p]['text'] for p in record['phrases'] if phrases[p]['type']=='noun')
            region_rows.append([f'clip{j:03d}',filename,record['key'][0],record['key'][1],record['key'][2],float(pts[frame]),
                                str((images/filename).resolve()),record['caption'],nouns,'','','','','',''])
        atomic(target/'progress.json',dict(completed=j+1,total=total,frames=len(region_rows)))
    decoded.close()
    headers=['片段编号','帧编号','视频UID','片段起点秒','片段终点秒','帧PTS秒','图像路径','弱描述','待核验物体短语',
             '物体可见_是/否/无法判断','实际交互_是/否/无法判断','区域JSON_224坐标_xyxy_可多框','描述与画面一致_是/否/无法判断','无法判断原因','备注']
    for rater in ('A','B'):
        with (target/f'区域核验_标注员{rater}.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f);writer.writerow(headers);writer.writerows(region_rows)
    proposals=read(ROOT/'reports/egoexo_relations_20261026/locked_review_candidates.json')['pairs']
    proposals=[p for p in proposals if p['kind']=='same_object_other_action'][:100]
    triplet_rows=[];mapping=[]
    for j,p in enumerate(proposals):
        a,b=p['proposed_positive'],p['proposed_negative']
        if rng.random()<.5:a,b=b,a
        rs=[records[i] for i in (p['query'],a,b)]
        triplet_rows.append([f'triplet{j:03d}',*[v for r in rs for v in [str(ROOT/'data/EgoExoLearn/videos'/(r['key'][0]+'.mp4')),r['key'][1],r['key'][2],r['caption']]],'','','','',''])
        mapping.append(dict(id=f'triplet{j:03d}',proposal=p,A_record=a,B_record=b,human_verified=False))
    th=['三元组编号']+[f'{who}_{field}' for who in ('查询','候选A','候选B') for field in ('视频','起点秒','终点秒','弱描述')]+[
        '是否同一物体_是/否/无法判断','动作差异可见_是/否/无法判断','唯一答案_A/B/均可/均不可/无法判断','可见证据','排除原因']
    for rater in ('A','B'):
        with (target/f'动作三元组_标注员{rater}.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f);writer.writerow(th);writer.writerows(triplet_rows)
    atomic(POST/'human_proposal_mapping.json',mapping)
    (target/'核验说明.txt').write_text('两位标注员独立填写各自A/B表；先不交换答案，不显示路由图、方法名或模型预测。\n区域坐标为导出224×224图像坐标，JSON可包含多个物体/交互区域；不可见、无法判断分别记录。\n描述与物体短语均来自弱叙述，需观看画面核实，不能默认正确。动作三元组需观看实际视频片段。\n保留两人原始表后再协商分歧，另存共识，不覆盖原表。后续报告可见性一致率及区域IoU。\n这些材料来自锁定的内部review视频；历史M初始化曾接触原holdout选择，并非从未接触的独立测试。\n所有人工字段留空；材料导出不代表人工核验完成。\n')
    atomic(target/'materials_complete.json',dict(passed=True,clips=total,frames=len(region_rows),action_triplets=len(triplet_rows),
           raters=2,all_human_fields_empty=True,human_review_complete=False,plan_sha256=digest(target/'material_plan.json')))
    print('MATERIALS_EXPORTED',total,len(region_rows),len(triplet_rows),flush=True)


if __name__=='__main__':main()
