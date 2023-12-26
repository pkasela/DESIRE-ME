import json
import logging
import os

import hydra
import torch
import tqdm
from indxr import Indxr
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModel, AutoTokenizer

from model.models import SpecialziedBiEncoder
from model.utils import seed_everything

from ranx import Run, Qrels, compare

logger = logging.getLogger(__name__)


def get_bert_rerank(data, model, doc_embedding, bm25_runs, id_to_index):
    bert_run = {}
    model.eval()
    for d in tqdm.tqdm(data, total=len(data)):
        with torch.no_grad():
            q_embedding = model.query_encoder_with_context([d['text']])
            
        bm25_docs = list(bm25_runs[d['_id']].keys())
        d_embeddings = doc_embedding[torch.tensor([int(id_to_index[x]) for x in bm25_docs])]
        bert_scores = torch.einsum('xy, ly -> x', d_embeddings, q_embedding)
        bert_run[d['_id']] = {doc_id: bert_scores[i].item() for i, doc_id in enumerate(bm25_docs)}
        
    return bert_run


def get_full_bert_rank(data, model, doc_embedding, id_to_index, k=100):
    bert_run = {}
    index_to_id = {ind: _id for _id, ind in id_to_index.items()}
    model.eval()
    for d in tqdm.tqdm(data, total=len(data)):
        with torch.no_grad():
            q_embedding = model.query_encoder_with_context([d['text']])
        
        bert_scores = torch.einsum('xy, ly -> x', doc_embedding, q_embedding)
        index_sorted = torch.argsort(bert_scores, descending=True)
        top_k = index_sorted[:k]
        bert_ids = [index_to_id[int(_id)] for _id in top_k]
        bert_scores = bert_scores[top_k]
        bert_run[d['_id']] = {doc_id: bert_scores[i].item() for i, doc_id in enumerate(bert_ids)}
        
        
    return bert_run
    
@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    os.makedirs(cfg.dataset.output_dir, exist_ok=True)
    os.makedirs(cfg.dataset.logs_dir, exist_ok=True)
    os.makedirs(cfg.dataset.model_dir, exist_ok=True)
    os.makedirs(cfg.dataset.runs_dir, exist_ok=True)
    
    logging_file = f"{cfg.model.init.doc_model.replace('/','_')}_testing.log"
    logging.basicConfig(
        filename=os.path.join(cfg.dataset.logs_dir, logging_file),
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )

    seed_everything(cfg.general.seed)

    with open(cfg.dataset.category_to_label, 'r') as f:
        category_to_label = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.init.tokenizer)
    doc_model = AutoModel.from_pretrained(cfg.model.init.doc_model)    
    model = SpecialziedBiEncoder(
        doc_model=doc_model,
        tokenizer=tokenizer,
        num_classes=len(category_to_label),
        normalize=cfg.model.init.normalize,
        specialized_mode=cfg.model.init.specialized_mode,
        pooling_mode=cfg.model.init.aggregation_mode,
        device=cfg.model.init.device
    )
    
    model.load_state_dict(torch.load(f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}.pt'))
    
    if cfg.testing.rerank:
        prefix = 'rerank'
    else:
        prefix = 'fullrank'
        
    prefix += '_' + cfg.model.init.specialized_mode
        
    doc_embedding = torch.load(f'{cfg.testing.embedding_dir}/{cfg.model.init.save_model}_fullrank.pt').to(cfg.model.init.device)
    
    with open(f'{cfg.testing.embedding_dir}/id_to_index_{cfg.model.init.save_model}_fullrank.json', 'r') as f:
        id_to_index = json.load(f)
    
    with open(cfg.testing.bm25_run_path, 'r') as f:
        bm25_run = json.load(f)
    
    data = Indxr(cfg.testing.query_path, key_id='_id')
    if cfg.testing.rerank:
        bert_run = get_bert_rerank(data, model, doc_embedding, bm25_run, id_to_index)
    else:
        bert_run = get_full_bert_rank(data, model, doc_embedding, id_to_index, 100)
        
    
    with open(f'{cfg.dataset.runs_dir}/{cfg.model.init.save_model}_{prefix}.json', 'w') as f:
        json.dump(bert_run, f)
        
        
    ranx_qrels = Qrels.from_file(cfg.testing.qrels_path)
    
    if cfg.testing.rerank:
        ranx_run = Run(bert_run, 'ReRanker')
        ranx_bm25_run = Run(bm25_run, name='BM25')
        models = [ranx_bm25_run, ranx_run]
    else:
        ranx_run = Run(bert_run, 'FullRun')
        models = [ranx_run]
    evaluation_report = compare(ranx_qrels, models, ['map@100', 'mrr@10', 'recall@100', 'ndcg@10', 'precision@1', 'ndcg@3'])
    print(evaluation_report)
    logging.info(f"Results for {cfg.model.init.save_model}_{prefix}.json:\n{evaluation_report}")

if __name__ == '__main__':
    main()
