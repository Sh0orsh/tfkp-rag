import argparse,csv,json,subprocess,sys,statistics,time
from pathlib import Path

MODES=[('no_context','no_context'),('single_search','baseline'),('final','final'),('ablation_no_dense','bm25_only'),('ablation_no_bm25','dense_only')]
def read(p):return [json.loads(s) for s in open(p,encoding='utf-8') if s.strip()]
def main():
 p=argparse.ArgumentParser()
 for a in ['index','questions','gold','config']:p.add_argument('--'+a,required=True)
 p.add_argument('--output-dir',default='experiments_output');a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 gold={x['id']:x for x in read(a.gold)};rows=[]
 for name,mode in MODES:
  ap=out/(name+'_answers.jsonl');tp=out/(name+'_trace.jsonl')
  subprocess.run([sys.executable,str(Path(__file__).resolve().parent/'predict.py'),'--index',a.index,'--questions',a.questions,'--output',str(ap),'--trace',str(tp),'--config',a.config,'--mode',mode],check=True)
  answers=read(ap);trace=read(tp);acc=sum(set(x['answer_ids'])==set(gold[x['id']]['answer_ids']) and x['abstain']==gold[x['id']]['abstain'] for x in answers)/len(answers)
  avg=lambda k:statistics.mean(x[k] for x in trace)
  cost=statistics.mean(x['input_tokens']+4*x['output_tokens']+256*x['llm_calls'] for x in trace)
  rows.append(dict(name=name,answer_accuracy=round(acc,4),Q='',mean_input_tokens=round(avg('input_tokens'),2),mean_output_tokens=round(avg('output_tokens'),2),mean_llm_calls=round(avg('llm_calls'),2),mean_cost=round(cost,2),median_latency_ms=round(statistics.median(x['elapsed_ms'] for x in trace),2)))
 with open(Path(__file__).resolve().parent/'experiments.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('Saved experiments.csv; Q requires category labels and citation F1 scoring')
if __name__=='__main__':main()
