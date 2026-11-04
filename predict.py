import argparse,json,os,re,time
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from local_llm import LocalQwen
from build_index import read,tok

SYSTEM='''Ты решаешь тест по ТФКП исключительно по приведённым фрагментам учебника. Выбери ВСЕ и только правильные варианты. Учитывай условия теорем, контрпримеры, ориентацию контуров, факториалы. Если исходные условия не определяют единственный требуемый ответ, откажись (abstain=true), даже если несколько значений возможны. Не путай недостаточность условий задачи с отсутствием найденной теоремы. Приводи не более 4 исходных block_id, которые действительно обосновывают вывод. Ответ строго JSON: {"answer_ids":[],"abstain":true,"rationale":"...","citations":[]}. Обоснование краткое, до 500 символов.'''
def parse(s):
    s=re.sub(r'```(?:json)?|```','',s).strip();m=re.search(r'\{.*\}',s,re.S)
    if not m:raise ValueError('No JSON')
    return json.loads(m.group())
def normalize(obj, q, allowed, mode="final"):
    ids = {x["id"] for x in q["options"]}
    ans = list(dict.fromkeys(x for x in obj.get("answer_ids", []) if x in ids))
    cit = list(dict.fromkeys(x for x in obj.get("citations", []) if x in allowed))[:4]
    abstain = bool(obj.get("abstain", False))

    if mode == "no_context":
        abstain = abstain or not ans
        cit = []
    else:
        abstain = abstain or not ans or not cit

    return {
        "id": q["id"],
        "answer_ids": [] if abstain else ans,
        "abstain": abstain,
        "rationale": str(obj.get("rationale") or "Недостаточно данных.")[:1200],
        "citations": [] if abstain else cit
    }
class Retriever:
    def __init__(self,index,cfg):
        self.chunks=json.load(open(Path(index)/'chunks.json',encoding='utf-8'));self.cfg=cfg
        self.bm=BM25Okapi([tok(x['section']+' '+x['text']) for x in self.chunks]);self.emb=None;self.model=None
        if cfg['dense']:
            self.emb=np.load(Path(index)/'embeddings.npy',mmap_mode='r');self.model=SentenceTransformer(cfg['embedding_model'],revision=cfg.get('embedding_revision') or None,device='cpu')
    def search(self,q,mode='final'):
        if mode=='no_context':return []
        query=q['question']+' '+' '.join(x['text'] for x in q['options']);n=len(self.chunks);top=min(n,60)
        rrf={}
        if mode!='dense_only':
            scores=self.bm.get_scores(tok(query));rank=np.argsort(-scores)[:top];rrf={int(i):1/(60+j) for j,i in enumerate(rank,1)}
        if self.emb is not None and mode not in ('bm25_only','baseline'):
            vec=self.model.encode(["query: "+query],normalize_embeddings=True)[0];sim=np.asarray(self.emb@vec);rank=np.argsort(-sim)[:top]
            for j,i in enumerate(rank,1):rrf[int(i)]=rrf.get(int(i),0)+1/(60+j)
        ordered=sorted(rrf,key=rrf.get,reverse=True);result=[];seen=set()
        for i in ordered:
            c=self.chunks[i];bid=c['block_id']
            if bid in seen:continue
            seen.add(bid);result.append(c)
            if len(result)>=self.cfg['top_k']:break
        return result

def prompt(q,blocks,limit):
    question=q['question']+'\n'+'\n'.join(f"{x['id']}. {x['text']}" for x in q['options'])
    context=[];allowed=[];used=0
    for b in blocks:
        line=f"[block_id={b['block_id']}; page={b['page']}; section={b['section']}]\n{b['text']}";length=len(line)//3
        if used+length>limit:continue
        context.append(line);allowed.append(b['block_id']);used+=length
    return question+'\n\nФрагменты учебника:\n'+'\n\n'.join(context),allowed

def run(index,questions,output,trace,config,mode='final'):
    cfg=json.load(open(config));ret=Retriever(index,cfg) if mode!='no_context' else None;client=LocalQwen(cfg['llm_model'],cfg['max_tokens']);answers=[];traces=[]
    for q in read(questions):
        t=time.perf_counter();blocks=ret.search(q,mode) if ret is not None else [];user,allowed=prompt(q,blocks,cfg['max_context_chars']);calls=0;tin=0;tout=0
        messages=[{'role':'system','content':SYSTEM},{'role':'user','content':user}]
        try:
            content,tin,tout=client.generate(messages);calls=1
            parsed=parse(content);answer = normalize(parsed, q, allowed, mode)
            if mode=='no_context':answer['citations']=[]
        except Exception as exc:
            answer={'id':q['id'],'answer_ids':[],'abstain':True,'rationale':'Ошибка генерации или формата: '+str(exc)[:160],'citations':[]}
        answers.append(answer);traces.append({'id':q['id'],'llm_calls':calls,'input_tokens':tin,'output_tokens':tout,'elapsed_ms':round((time.perf_counter()-t)*1000),'retrieved_block_ids':allowed})
        print(q['id'],answer['answer_ids'],'abstain',answer['abstain'],flush=True)
    for path,items in [(output,answers),(trace,traces)]:
        with open(path,'w',encoding='utf-8') as f:
            for item in items:f.write(json.dumps(item,ensure_ascii=False)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for arg in ('index','questions','output','trace','config'):p.add_argument('--'+arg,required=True)
    p.add_argument('--mode',default='final',choices=['final','baseline','bm25_only','no_context','dense_only']);a=p.parse_args();run(a.index,a.questions,a.output,a.trace,a.config,a.mode)
