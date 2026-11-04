import argparse,json,re,time
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

def read(path):
    with open(path,encoding='utf-8') as f:return [json.loads(x) for x in f if x.strip()]
def tok(s):return re.findall(r'[a-zа-яё0-9]+',s.lower().replace('ё','е'))
def main(corpus,output,config):
    t=time.time();cfg=json.load(open(config));out=Path(output);out.mkdir(parents=True,exist_ok=True)
    blocks=read(corpus);chunks=[];size=cfg['chunk_size'];overlap=cfg['overlap']
    for b in blocks:
        words=b['text'].split();step=size-overlap
        for start in range(0,max(1,len(words)),step):
            part=' '.join(words[start:start+size]);chunks.append({'block_id':b['block_id'],'page':b['page'],'section':b['section'],'text':part})
            if start+size>=len(words):break
    with open(out/'chunks.json','w',encoding='utf-8') as f:json.dump(chunks,f,ensure_ascii=False)
    if cfg['dense']:
        model=SentenceTransformer(cfg['embedding_model'],revision=cfg.get('embedding_revision') or None,device='cpu')
        embeddings=model.encode([c['section']+' '+c['text'] for c in chunks],batch_size=32,normalize_embeddings=True,show_progress_bar=True)
        np.save(out/'embeddings.npy',embeddings.astype('float32'))
    print('Chunks:',len(chunks),'Build seconds:',round(time.time()-t,1))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',required=True);p.add_argument('--output',required=True);p.add_argument('--config',required=True);a=p.parse_args();main(a.corpus,a.output,a.config)
