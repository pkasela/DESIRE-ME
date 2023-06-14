import json
import logging
import os

import hydra
import torch
import tqdm
from indxr import Indxr
from omegaconf import DictConfig
from transformers import AutoModel, AutoTokenizer

from model.models import BiEncoder, BiEncoderCLS
from model.utils import seed_everything

logger = logging.getLogger(__name__)


    
@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logging_file = "create_embedding.log"
    logging.basicConfig(
        filename=os.path.join(cfg.general.logs_dir, logging_file),
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )

    
    seed_everything(cfg.general.seed)
    
    corpus = Indxr(cfg.test.corpus_path, key_id='_id')
    with open(cfg.general.category_to_label, 'r') as f:
        category_to_label = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.init.tokenizer)
    doc_model = AutoModel.from_pretrained(cfg.model.init.doc_model)
    model = BiEncoderCLS(
        doc_model=doc_model,
        tokenizer=tokenizer,
        num_classes=len(category_to_label),
        device=cfg.model.init.device,
        mode=cfg.model.init.aggregation_mode
    )
    logging.info(f'Loading model from {cfg.model.init.doc_model.replace("/","_")}_best.pt')
    if os.path.exists(f'{cfg.general.model_dir}/{cfg.model.init.doc_model.replace("/","_")}_best.pt'):
        model.load_state_dict(torch.load(f'{cfg.general.model_dir}/{cfg.model.init.doc_model.replace("/","_")}_best.pt'))
    else:
        logging.info('New model requested, creating new checkpoint')
        torch.save(model.state_dict(), f'{cfg.general.model_dir}/{cfg.model.init.doc_model.replace("/","_")}_best.pt')

    
    index = 0
    texts = []
    id_to_index = {}
    with open(cfg.test.bm25_run_path, 'r') as f:
        bm25_run = json.load(f)
    
    # important_docs = []
    # for run in bm25_run:
    #     important_docs.extend(list(bm25_run[run]))
    # important_docs = set(important_docs)
    embedding_matrix = torch.zeros(len(corpus), cfg.model.init.embedding_size).float()
    for doc in tqdm.tqdm(corpus):
        if True: #doc['_id'] in important_docs:
            id_to_index[doc['_id']] = index
            index += 1
            texts.append(doc['title'] + '. ' + doc['text'])
            if len(texts) == cfg.training.batch_size:
                with torch.no_grad():
                    embedding_matrix[index - len(texts) : index] = model.doc_encoder(texts).cpu()
                texts = []
    if texts:
        with torch.no_grad():
            embedding_matrix[index - len(texts) : index] = model.doc_encoder(texts).cpu()
            
    logging.info(f'Embedded {index} documents. Saving embedding matrix in folder {cfg.test.embedding_dir}.')
    os.makedirs(cfg.test.embedding_dir, exist_ok=True)
    torch.save(embedding_matrix, f'{cfg.test.embedding_dir}/{cfg.model.init.doc_model.replace("/","_")}.pt')
        
    logging.info('Saving id_to_index file.')
    with open(f'{cfg.test.embedding_dir}/id_to_index_{cfg.model.init.doc_model.replace("/","_")}.json', 'w') as f:
        json.dump(id_to_index, f)
    
if __name__ == '__main__':
    main()
    