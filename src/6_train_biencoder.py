import json
import logging
import os

import hydra
import numpy as np
import pandas as pd
import torch
import tqdm
from omegaconf import DictConfig
from torch import save, load
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer

from dataloader.dataloader import LoadTrainNQData
from model.loss import MultipleRankingLoss, MultipleRankingLossBiEncoder
from model.models import SpecialziedBiEncoder, BiEncoder
from model.utils import seed_everything

logger = logging.getLogger(__name__)

def train(train_data, model, optimizer, loss_fn, batch_size, epoch, device):
    """
    Training function

    Args:
        train_data (torch.utils.data.Dataset): the training data
        model (): the neural language model
        optimizer (): an optimizer for backpropagation
        loss_fn (): a loss function which will used to train the model
        batch_size (int): batch size
        epoch (int): epoch number, used for logging and progress bar
        device (str): 'cpu' or 'cuda'

    Returns:
        average_loss (float): returns the average training loss value for the current epoch
    """
    losses = []
    
    data = torch.utils.data.DataLoader(
        train_data, 
        batch_size=batch_size,
        shuffle=True
    )
    train_data = tqdm.tqdm(data)
    optimizer.zero_grad()
    for _, batch in enumerate(train_data):
        with torch.cuda.amp.autocast():
            output = model((batch['question'], batch['pos_text']))
            class_label = torch.ones(batch['pos_category'].shape[0], 1).to(device)
            loss_val = loss_fn(
                # class_label, output[0],
                output[0], output[1]
            )
        loss_val.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss_val.cpu().detach().item())
        
        average_loss = np.mean(losses)
        
        train_data.set_description("TRAIN EPOCH {:3d} Current loss {:.2e}, Average {:.2e}".format(epoch, loss_val, average_loss))

    return average_loss

    
def validate(val_data, model, loss_fn, batch_size, epoch, device):
    """
    Training function

    Args:
        val_data (torch.utils.data.Dataset): the validation data
        model (): the neural language model
        loss_fn (): a loss function which will used to compute the validation loss
        batch_size (int): batch size
        epoch (int): epoch number, used for logging and progress bar
        device (str): 'cpu' or 'cuda'

    Returns:
        average_loss (float): returns the average validation loss value for the current epoch
    """
    
    losses = []
    sim_accuracy = []
    data = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=True
    )
    val_data = tqdm.tqdm(data)
    for _, batch in enumerate(val_data):
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output = model.val_forward((batch['question'], batch['pos_text']))
                class_label = torch.ones(batch['pos_category'].shape[0], 1).to(device)
                loss_val, sim_correct = loss_fn.val_forward(
                    # class_label, output[0],
                    output[0], output[1]
                )
                sim_accuracy.extend(sim_correct.tolist())

        losses.append(loss_val.cpu().detach().item())
        average_loss = np.mean(losses)
        average_sim_accuracy = np.mean(sim_accuracy)
        
        val_data.set_description("VAL EPOCH {:3d} Sim Accuracy {}, Average Loss {:.2e}".format(epoch, round(average_sim_accuracy*100,2), average_loss))

    return average_loss


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    os.makedirs(cfg.dataset.output_dir, exist_ok=True)
    os.makedirs(cfg.dataset.logs_dir, exist_ok=True)
    os.makedirs(cfg.dataset.model_dir, exist_ok=True)
    os.makedirs(cfg.dataset.runs_dir, exist_ok=True)
    
    logging_file = f"{cfg.model.init.doc_model.replace('/','_')}_training_biencoder.log"
    logging.basicConfig(filename=os.path.join(cfg.dataset.logs_dir, logging_file),
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO
                        )

    seed_everything(cfg.general.seed)

    with open(cfg.dataset.category_to_label, 'r') as f:
        category_to_label = json.load(f)

    for cat in category_to_label:
        category_to_label[cat] = 0

    qrel_df = pd.read_csv(cfg.dataset.qrels_path, sep='\t')
    qrels = {}
        
    for index, row in qrel_df.iterrows():
        q_id = str(row['query-id']) 
        
        if not q_id in qrels:
            qrels[q_id] = {}
        
        qrels[q_id][str(row['corpus-id'])] = row['score']

    data = LoadTrainNQData(
        cfg.dataset.query_path, 
        cfg.dataset.corpus_path, 
        qrels, 
        category_to_label
    )

    val_split = cfg.dataset.val_split
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
        # num_classes=1,
        normalize=cfg.model.init.normalize,
        # specialized_mode='ones',
        pooling_mode=cfg.model.init.aggregation_mode,
        device=cfg.model.init.device
    )
    logging.info("Model: {}, lr: {:.2e}, batch_size: {}, epochs: {}".format(cfg.model.init.doc_model, cfg.training.lr, cfg.training.batch_size, cfg.training.max_epoch))
    logging.info("Normalize: {}, specialized mode: {}, pooling mode: {}".format(cfg.model.init.normalize, cfg.model.init.specialized_mode, cfg.model.init.aggregation_mode))
    loss_fn = MultipleRankingLossBiEncoder(device=cfg.model.init.device, temperature=cfg.model.init.temperature)

    batch_size = cfg.training.batch_size
    max_epoch = cfg.training.max_epoch
    
    
    if cfg.model.continue_train:
        logging.info('Loading previous best model to continue training')
        model.load_state_dict(load(f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}.pt'))
        best_val_loss = validate(val_data, model, loss_fn, batch_size, 0, cfg.model.init.device)
        logging.info("VAL EPOCH: {}, Average Loss: {:.5e}".format('prev best', best_val_loss))
        
    
    else:
        best_val_loss = 999
    
    optimizer = AdamW(model.parameters(), lr=cfg.training.lr)
    
    for epoch in tqdm.tqdm(range(max_epoch)):
        model.train()
        average_loss = train(train_data, model, optimizer, loss_fn, batch_size, epoch + 1, cfg.model.init.device)
        logging.info("TRAIN EPOCH: {:3d}, Average Loss: {:.5e}".format(epoch + 1, average_loss))
        
        model.eval()
        val_loss = validate(val_data, model, loss_fn, batch_size, epoch + 1, cfg.model.init.device)
        logging.info("VAL EPOCH: {:3d}, Average Loss: {:.5e}".format(epoch + 1, val_loss))
        
        if val_loss < best_val_loss:
            logging.info(f'Found new best model on epoch: {epoch + 1}, new best validation loss {val_loss}')
            best_val_loss = val_loss
            logging.info(f'saving model checkpoint at epoch {epoch + 1}')
            save(model.state_dict(), f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}.pt')


if __name__ == '__main__':
    main()
