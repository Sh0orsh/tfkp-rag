"""Recompute measured accuracy and citation F1; category-balanced Q only when gold categories are supplied."""
import argparse,csv,json,statistics
from pathlib import Path

def read(p):return [json.loads(s) for s in open(p,encoding='utf-8') if s.strip()]
def gold_sets(g):
    sets=g.get('acceptable_citation_sets')
    if isinstance(sets,list) and sets and all(isinstance(x,list) for x in sets):return [set(x) for x in sets]
    c=g.get('citations',[])
    return [set(c)] if c else []
def f1(a,b):
    if not a and not b:return 1.
    return 2*len(a&b)/(len(a)+len(b)) if a or b else 0.
def evaluate(gold,answers,traces):
    gs={x['id']:x for x in read(gold)}; ans=read(answers);ts={x['id']:x for x in read(traces)}
    if not ans:raise ValueError('Empty answers')
    correct=[];scores=[];cats={};valid_categories=True
    for a in ans:
        g=gs[a['id']];ok=set(a['answer_ids'])==set(g['answer_ids']) and bool(a['abstain'])==bool(g['abstain']);correct.append(int(ok))
        if g['abstain']:score=float(ok)
        else:
            fs=gold_sets(g);F=max((f1(set(a.get('citations',[])),b) for b in fs),default=0.)
            score=float(ok)*(0.8+0.2*F)
        scores.append(score);cat=g.get('category')
        if cat is None:valid_categories=False
        else:cats.setdefault(str(cat),[]).append(score)
    traces=[ts[a['id']] for a in ans]
    mean=lambda k:statistics.mean(t[k] for t in traces)
    return {'answer_accuracy':round(statistics.mean(correct),4),'Q':round(statistics.mean(statistics.mean(v) for v in cats.values()),4) if valid_categories and len(cats)==4 else '', 'mean_input_tokens':round(mean('input_tokens'),2),'mean_output_tokens':round(mean('output_tokens'),2),'mean_llm_calls':round(mean('llm_calls'),2),'mean_cost':round(statistics.mean(t['input_tokens']+4*t['output_tokens']+256*t['llm_calls'] for t in traces),2),'median_latency_ms':round(statistics.median(t['elapsed_ms'] for t in traces),2)}
def main():
    p=argparse.ArgumentParser();p.add_argument('--gold',required=True);p.add_argument('--results-dir',default='experiments_output');p.add_argument('--output',default='experiments_recomputed.csv');a=p.parse_args()
    modes=['no_context','single_search','final','ablation_no_dense','ablation_no_bm25'];rows=[]
    for mode in modes:
        ap=Path(a.results_dir)/(mode+'_answers.jsonl');tp=Path(a.results_dir)/(mode+'_trace.jsonl')
        if not ap.exists() or not tp.exists():raise FileNotFoundError(f'{ap} or {tp}')
        r={'name':mode,**evaluate(a.gold,ap,tp)}
        if mode=='no_context':r['Q']=''
        rows.append(r)
    with open(a.output,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('Saved:',a.output,'(Q blank if four gold categories unavailable)')
if __name__=='__main__':main()
