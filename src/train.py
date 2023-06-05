import hydra
from omegaconf import DictConfig, OmegaConf

import torch
from indxr import Indxr
import random
import json
import os

from transformers import AutoModel, AutoTokenizer
from torch.optim import AdamW
from torch import save, load
import tqdm
import numpy as np
import pandas as pd
import logging

from model.utils import seed_everything
from model.models import BiEncoder
from model.loss import TripletMarginClassLoss

from dataloader.dataloader import LoadTrainNQData, in_batch_negative_collate_fn


logger = logging.getLogger(__name__)

def train(train_data, model, optimizer, loss_fn, batch_size, epoch, device):
    losses = []
    data = torch.utils.data.DataLoader(
        train_data, 
        collate_fn=in_batch_negative_collate_fn, 
        batch_size=batch_size, 
        shuffle=True
    )
    train_data = tqdm.tqdm(data)
    optimizer.zero_grad()
    for step, batch in enumerate(train_data):
        with torch.cuda.amp.autocast():
            output = model.forward_random_neg((batch['question'], batch['pos_text'], batch['neg_text']))
            loss_val = loss_fn(
                torch.tensor(batch['pos_category']).to(device), output[0], 
                output[1], output[2], output[3]
            )
        
        loss_val.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss_val.cpu().detach().item())
        average_loss = np.mean(losses)
        train_data.set_description("TRAIN EPOCH {:3d} Current loss {:.2e}, Average {:.2e}".format(epoch, loss_val, average_loss))

    return average_loss    

    
def validate(val_data, model, optimizer, loss_fn, batch_size, epoch, device):
    losses = []
    accuracy = []
    data = torch.utils.data.DataLoader(
        val_data, 
        collate_fn=in_batch_negative_collate_fn, 
        batch_size=batch_size, 
        shuffle=True
    )
    val_data = tqdm.tqdm(data)
    optimizer.zero_grad()
    for step, batch in enumerate(val_data):
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output = model.forward_random_neg((batch['question'], batch['pos_text'], batch['neg_text']))
                loss_val = loss_fn(
                    torch.tensor(batch['pos_category']).to(device), output[0], 
                    output[1], output[2], output[3]
                )
                predictions = output[0].argmax(dim=1, keepdim=True).squeeze()
                correct = (predictions == torch.tensor(batch['pos_category']).to(device))
                accuracy.extend(correct.tolist())

        losses.append(loss_val.cpu().detach().item())
        average_loss = np.mean(losses)
        average_accuracy = np.mean(accuracy)
        val_data.set_description("VAL EPOCH {:3d} Average Accuracy {:.2e}, Average Loss {:.2e}".format(epoch, round(average_accuracy*100,2), average_loss))

    return average_loss    


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    os.makedirs(cfg.general.output_dir, exist_ok=True)
    os.makedirs(cfg.general.logs_dir, exist_ok=True)
    os.makedirs(cfg.general.model_dir, exist_ok=True)
    
    logging_file = f"training.log"
    logging.basicConfig(filename=os.path.join(cfg.general.logs_dir, logging_file),
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO
                        )
    
    seed_everything(cfg.general.seed)

    with open(cfg.general.category_to_label, 'r') as f:
        category_to_label = json.load(f)
    
    qrel_df = pd.read_csv(cfg.general.qrels_path, sep='\t')
    qrels = {}

    for index, row in qrel_df.iterrows():
        qrels[row['query-id']] = {row['corpus-id']: row['score']}

    data = LoadTrainNQData(
        cfg.general.query_path, 
        cfg.general.corpus_path, 
        qrels, 
        category_to_label
    )
    
    val_split = cfg.general.val_split
    if val_split < 1:
        train_split = 1 - val_split
    else:
        train_split = len(data) - val_split
    
    train_data, val_data = torch.utils.data.random_split(
        data, 
        [train_split, val_split]
    )
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.init.tokenizer)
    doc_model = AutoModel.from_pretrained(cfg.model.init.doc_model)
    model = BiEncoder(
        doc_model=doc_model,
        tokenizer=tokenizer, 
        num_classes=len(category_to_label), 
        device=cfg.model.init.device,
        mode=cfg.model.init.aggregation_mode
    )    
    loss_fn = TripletMarginClassLoss()
    
    batch_size = cfg.training.batch_size
    max_epoch = cfg.training.max_epoch
    optimizer = AdamW(model.parameters(), lr=cfg.training.lr)
    
    best_val_loss = 999
    for epoch in tqdm.tqdm(range(max_epoch)):
        model.train()
        average_loss = train(train_data, model, optimizer, loss_fn, batch_size, epoch, cfg.model.init.device)
        logging.info("TRAIN EPOCH: {:3d}, Average Loss: {:.2e}".format(epoch, average_loss))
        
        model.eval()
        val_loss = validate(val_data, model, optimizer, loss_fn, batch_size, epoch, cfg.model.init.device)
        logging.info("VAL EPOCH: {:3d}, Average Loss: {:.2e}".format(epoch, val_loss))
        
        if val_loss < best_val_loss:
            logging.info(f'Found new best model on epoch: {epoch + 1}, new best validation loss {val_loss}')
            best_val_loss = val_loss
            logging.info(f'saving model checkpoint at epoch {epoch + 1}')
            save(model.state_dict(), f'{cfg.general.model_dir}/{cfg.model.init.doc_model}_{epoch+1}.pt')


if __name__ == '__main__':
    main()