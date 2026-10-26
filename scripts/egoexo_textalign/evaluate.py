import json
import numpy as np,torch
from common import ROOT,EXP,REP,ARMS,SEEDS,read,atomic,digest,cool
from model import VideoTail
from data import Prefixes

def main():
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.6)
    for p,h in read(EXP/'protocol.json')['sources'].items():assert digest(p)==h,p
    plan=read(EXP/'plan.json');ep=np.load(EXP/'val_episodes.npz');ids=np.unique(ep['indices']);lookup={int(i):j for j,i in enumerate(ids)};index=np.vectorize(lookup.get)(ep['indices']);truth=ep['truth'];a=Prefixes();scores={};cosines={};details={}
    configurations=[('initial',None,None)]+[(arm,seed,EXP/'heads'/arm/f'seed{seed}') for arm in ARMS for seed in SEEDS]
    for arm,seed,folder in configurations:
        m=VideoTail().eval()
        if folder:
            done=read(folder/'complete.json');assert digest(folder/'best.pt')==done['checkpoint_sha256'];state=torch.load(folder/'best.pt',map_location='cpu');assert state['manifest_sha256']==digest(folder/'manifest.json');result=m.load_state_dict(state['updates'],strict=False);assert not result.unexpected_keys
            expected=set(state['updates']);assert expected==set(m.updates());details[f'{arm}_seed{seed}']=dict(epoch=state['epoch'],held_loss=done['best_held_loss'])
        features=[]
        with torch.inference_mode():
            for b in range(0,len(ids),64):
                cool();x,mask=a.batch(ids[b:b+64],False);g,_,_=m(x.float(),False);features.append(g.cpu().numpy())
        f=np.concatenate(features);assert np.isfinite(f).all();dot=np.einsum('nd,njd->nj',f[index[:,0]],f[index[:,1:]]);norm=f/np.linalg.norm(f,axis=1,keepdims=True);cos=np.einsum('nd,njd->nj',norm[index[:,0]],norm[index[:,1:]])
        name='initial' if seed is None else f'{arm}_seed{seed}';scores[name]=dot;cosines[name]=cos;del m;torch.cuda.empty_cache()
        atomic(EXP/'evaluation_progress.json',dict(completed=name,updated_results=len(scores),total=13))
    np.savez_compressed(EXP/'val_scores.npz',uids=ep['uids'],truth=truth,**scores);np.savez_compressed(EXP/'val_cosine_scores.npz',uids=ep['uids'],truth=truth,**cosines)
    metrics={};comparisons={}
    for arm in ARMS:
        counts=[int((scores[f'{arm}_seed{s}'].argmax(1)==truth).sum()) for s in SEEDS];cc=[int((cosines[f'{arm}_seed{s}'].argmax(1)==truth).sum()) for s in SEEDS];metrics[arm]=dict(correct_counts=counts,mean_accuracy=float(np.mean(counts)/8),cosine_correct_counts=cc,cosine_mean_accuracy=float(np.mean(cc)/8))
    for left,right in [('B','A'),('C','A'),('C','B'),('C','D')]:
        changes=[]
        for seed in SEEDS:
            good=scores[f'{left}_seed{seed}'].argmax(1)==truth;old=scores[f'{right}_seed{seed}'].argmax(1)==truth;changes.append(dict(seed=seed,repaired=int((good&~old).sum()),broken=int((~good&old).sum()),net=int(good.sum()-old.sum())))
        comparisons[left+'_vs_'+right]=dict(delta_pp=metrics[left]['mean_accuracy']-metrics[right]['mean_accuracy'],changes=changes)
    initial=dict(correct=int((scores['initial'].argmax(1)==truth).sum()),accuracy=float(100*(scores['initial'].argmax(1)==truth).mean()),cosine_accuracy=float(100*(cosines['initial'].argmax(1)==truth).mean()))
    report=dict(complete=True,validation_n=800,primary_metric='raw dot, fixed before training',initial=initial,metrics=metrics,comparisons=comparisons,checkpoints=details,TEST_evaluated=False,frames=4,limitations=['Continued local historical checkpoint, not a clean reimplementation of paper pretraining.','Fixed sampled frames and fixed spatial transform; last2encoderblocks really update; first10cached.','Labels are weak taxonomy phrases supported by captions; visibility and temporal grounding remain unverified.','Caption teacher is existing frozen text tower, not a new LLM.','Internal holdout uses sentence loss; official VAL only evaluated after fits.','Cosine is diagnostic, not used to retrospectively replace primary raw-dot metric.'])
    atomic(REP/'results.json',report);lines=['EgoExo 细粒度文本后训练：A/B/C/D验证集结果','',f'缓存同条件初始模型：{initial["accuracy"]:.3f}%（点积）；{initial["cosine_accuracy"]:.3f}%（余弦）。']
    names={'A':'整句对齐','B':'整句＋全局短语','C':'整句＋全局短语＋局部短语对齐','D':'与C相同但局部短语对应打乱'}
    for arm,m in metrics.items():lines.append(f'{arm} {names[arm]}：{m["mean_accuracy"]:.3f}%，三个种子正确数{m["correct_counts"]}/800；余弦诊断{m["cosine_mean_accuracy"]:.3f}%。')
    lines+=['',json.dumps(comparisons,ensure_ascii=False,indent=2),'','本轮更新视频编码器最后两层、最终归一化和投影；冻结文本塔。仅训练/内部留出/官方验证，未评测TEST。','240条人工核验队列仍待人工核验，本轮明确按弱监督解释。']
    (REP/'结果阅读版.txt').write_text('\n'.join(lines)+'\n');atomic(EXP/'complete.json',dict(passed=True,TEST_evaluated=False,report_sha256=digest(REP/'results.json')));print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
