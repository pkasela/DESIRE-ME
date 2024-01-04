import json
import logging
import os

import hydra
import torch
import tqdm
from indxr import Indxr
from omegaconf import DictConfig
from transformers import AutoModel, AutoTokenizer

from model.models import SpecialziedBiEncoder
from model.utils import seed_everything

logger = logging.getLogger(__name__)


    
@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    os.makedirs(cfg.dataset.output_dir, exist_ok=True)
    os.makedirs(cfg.dataset.logs_dir, exist_ok=True)
    os.makedirs(cfg.dataset.model_dir, exist_ok=True)
    os.makedirs(cfg.dataset.runs_dir, exist_ok=True)
    
    logging_file = f"{cfg.model.init.doc_model.replace('/','_')}_create_embedding.log"
    logging.basicConfig(
        filename=os.path.join(cfg.dataset.logs_dir, logging_file),
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )

    
    seed_everything(cfg.general.seed)
    
    corpus = Indxr(cfg.testing.corpus_path, key_id='_id')
    corpus = sorted(corpus, key=lambda k: len(k.get("title", "") + k.get("text", "")), reverse=True)
    with open(cfg.dataset.category_to_label, 'r') as f:
        category_to_label = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.init.tokenizer)
    doc_model = AutoModel.from_pretrained(cfg.model.init.doc_model)    
    model = SpecialziedBiEncoder(
        doc_model=doc_model,
        tokenizer=tokenizer,
        num_classes=len(category_to_label),
        max_tokens=cfg.model.init.max_tokenizer_length,
        normalize=cfg.model.init.normalize,
        specialized_mode=cfg.model.init.specialized_mode,
        pooling_mode=cfg.model.init.aggregation_mode,
        device=cfg.model.init.device
    )
    logging.info(f'Loading model from {cfg.model.init.save_model}.pt')
    if os.path.exists(f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}.pt'):
        model.load_state_dict(torch.load(f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}.pt'))
    else:
        logging.info('New model CLS requested, creating new checkpoint')
        torch.save(model.state_dict(), f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}.pt')

    
    index = 0
    texts = []
    id_to_index = {}
    
    model.eval()
    embedding_matrix = torch.zeros(len(corpus), cfg.model.init.embedding_size).float()
    for doc in tqdm.tqdm(corpus):
        
        id_to_index[doc['_id']] = index
        index += 1
        texts.append(doc.get('title','').lower() + ' ' + doc['text'].lower())
        if len(texts) == cfg.training.batch_size:
            with torch.no_grad():
                embedding_matrix[index - len(texts) : index] = model.doc_encoder(texts).cpu()
            texts = []
    if texts:
        with torch.no_grad():
            embedding_matrix[index - len(texts) : index] = model.doc_encoder(texts).cpu()
            
    
    prefix = 'fullrank'
    logging.info(f'Embedded {index} documents. Saving embedding matrix in folder {cfg.testing.embedding_dir}.')
    os.makedirs(cfg.testing.embedding_dir, exist_ok=True)
    torch.save(embedding_matrix, f'{cfg.testing.embedding_dir}/{cfg.model.init.save_model}_{prefix}.pt')
        
    logging.info('Saving id_to_index file.')
    with open(f'{cfg.testing.embedding_dir}/id_to_index_{cfg.model.init.save_model}_{prefix}.json', 'w') as f:
        json.dump(id_to_index, f)
    
if __name__ == '__main__':
    main()
    