import json
import logging
import os

import hydra
import torch
import tqdm
from indxr import Indxr
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModel, AutoTokenizer

from model.models import BiEncoder, BiEncoderCLS
from model.utils import seed_everything

logger = logging.getLogger(__name__)


def get_bert_rank(data, model, doc_embedding, bm25_runs, id_to_index, query_specialize):
    bert_run = {}
    model.eval()
    for d in tqdm.tqdm(data, total=len(data)):
        with torch.no_grad():
            if query_specialize:
                q_embedding = model.query_encoder_with_context([d['text']])
            else:
                q_embedding = model.query_encoder([d['text']])
        
        bm25_docs = list(bm25_runs[d['_id']].keys())
        d_embeddings = doc_embedding[torch.tensor([int(id_to_index[x]) for x in bm25_docs])]
        bert_scores = torch.einsum('xy, ly -> x', d_embeddings, q_embedding)
        bert_run[d['_id']] = {doc_id: bert_scores[i].item() for i, doc_id in enumerate(bm25_docs)}
        
    return bert_run


def get_full_bert_rank(data, model, doc_embedding, id_to_index, query_specialize, k=100):
    bert_run = {}
    index_to_id = {ind: _id for _id, ind in id_to_index.items()}
    model.eval()
    # bert_bm25_run = {}
    for d in tqdm.tqdm(data, total=len(data)):
        with torch.no_grad():
            if query_specialize:
                q_embedding = model.query_encoder_with_context([d['text']])
            else:
                q_embedding = model.query_encoder([d['text']])
        
        bert_scores = torch.einsum('xy, ly -> x', doc_embedding, q_embedding)
        index_sorted = torch.argsort(bert_scores, descending=True)
        top_k = index_sorted[:k]
        bert_ids = [index_to_id[int(_id)] for _id in top_k]
        bert_scores = bert_scores[top_k]
        bert_run[d['_id']] = {doc_id: bert_scores[i].item() for i, doc_id in enumerate(bert_ids)}
        
        
    return bert_run
    
@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logging_file = "testing.log"
    logging.basicConfig(
        filename=os.path.join(cfg.general.logs_dir, logging_file),
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )

    seed_everything(cfg.general.seed)

    with open(cfg.general.category_to_label, 'r') as f:
        category_to_label = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.init.tokenizer)
    doc_model = AutoModel.from_pretrained(cfg.model.init.doc_model)
    model = BiEncoder(
        doc_model=doc_model,
        tokenizer=tokenizer,
        num_classes=len(category_to_label),
        device=cfg.model.init.device,
        mode=cfg.model.init.aggregation_mode
    )
    logging.info(f'Loading model from {cfg.model.init.doc_model.replace("/","_")}_best.pt')
    model.load_state_dict(torch.load(f'{cfg.general.model_dir}/{cfg.model.init.doc_model.replace("/","_")}_best.pt'))
    
    with open(cfg.test.bm25_run_path, 'r') as f:
        bm25_run = json.load(f)
    
    doc_embedding = torch.load(f'{cfg.test.embedding_dir}/{cfg.model.init.doc_model.replace("/","_")}.pt').to(cfg.model.init.device)
    
    with open(f'{cfg.test.embedding_dir}/id_to_index_{cfg.model.init.doc_model.replace("/","_")}.json', 'r') as f:
        id_to_index = json.load(f)
    
    data = Indxr(cfg.test.query_path, key_id='_id')
    # bert_run = get_bert_rank(data, model, doc_embedding, bm25_run, id_to_index, cfg.model.init.query_specialize)
    bert_run = get_full_bert_rank(data, model, doc_embedding, id_to_index, cfg.model.init.query_specialize, 100)
    with open(f'{cfg.test.data_dir}/{cfg.model.init.doc_model.replace("/","_")}.json', 'w') as f:
        json.dump(bert_run, f)


if __name__ == '__main__':
    main()