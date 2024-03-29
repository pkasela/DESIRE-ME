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
from model.loss import MultipleRankingLoss
from model.models import SpecialziedBiEncoder
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
    ce_losses = []
    triple_losses = []
    
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
            triple_loss, ce_loss, loss_val = loss_fn(
                batch['pos_category'].to(device), output[0],
                output[1], output[2]
            )
        loss_val.backward()
        # ce_loss.backward()
        # triple_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss_val.cpu().detach().item())
        ce_losses.append(ce_loss.cpu().detach().item())
        triple_losses.append(triple_loss.cpu().detach().item())
        
        average_loss = np.mean(losses)
        average_ce_loss = np.mean(ce_losses)
        average_triple_loss = np.mean(triple_losses)
        
        train_data.set_description("TRAIN EPOCH {:3d} Current loss {:.2e}, Average {:.2e} CE Loss {:.2e}, T Loss {:.2e}".format(epoch, loss_val, average_loss, average_ce_loss, average_triple_loss))

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
    ce_losses = []
    triple_losses = []
    accuracy = []
    recalls = []
    precisions = []
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
                triple_loss, ce_loss, loss_val, sim_correct = loss_fn.val_forward(
                    batch['pos_category'].to(device), output[0],
                    output[1], output[2]
                )
                predictions = torch.sigmoid(output[0]) > .5
                correct = (predictions) == batch['pos_category'].to(device)
                recall = (correct * batch['pos_category'].to(device)).sum() / batch['pos_category'].to(device).sum()
                recalls.append(recall.detach().cpu().item())
                precision = predictions.sum() if predictions.sum().int() == 0 else (correct * predictions).sum() / predictions.sum()
                precisions.append(precision.detach().cpu().item())
                accuracy.extend([l for c in correct.tolist() for l in c])
                sim_accuracy.extend(sim_correct.tolist())

        losses.append(loss_val.cpu().detach().item())
        ce_losses.append(ce_loss.cpu().detach().item())
        triple_losses.append(triple_loss.cpu().detach().item())
        average_loss = np.mean(losses)
        average_ce_loss = np.mean(ce_losses)
        average_triple_loss = np.mean(triple_losses)
        
        # average_accuracy = np.mean(accuracy)
        average_precision = np.mean(precisions)
        average_recall = np.mean(recalls)
        average_sim_accuracy = np.mean(sim_accuracy)
        val_data.set_description("VAL EPOCH {:3d} Average Precision {} Average Recall {} Sim Accuracy {}, Average Loss {:.2e}, CE Loss {:.2e}, T Loss {:.2e}".format(epoch, round(average_precision*100,2), round(average_recall*100,2), round(average_sim_accuracy*100,2), average_loss, average_ce_loss, average_triple_loss))

    return average_loss, average_precision, average_recall


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    os.makedirs(cfg.dataset.output_dir, exist_ok=True)
    os.makedirs(cfg.dataset.logs_dir, exist_ok=True)
    os.makedirs(cfg.dataset.model_dir, exist_ok=True)
    os.makedirs(cfg.dataset.runs_dir, exist_ok=True)
    
    logging_file = f"{cfg.model.init.doc_model.replace('/','_')}_training.log"
    logging.basicConfig(filename=os.path.join(cfg.dataset.logs_dir, logging_file),
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO
                        )

    seed_everything(cfg.general.seed)

    with open(cfg.dataset.category_to_label, 'r') as f:
        category_to_label = json.load(f)

    qrel_df = pd.read_csv(cfg.dataset.qrels_path, sep='\t')
    qrels = {}

    # for _, row in qrel_df.iterrows():
    #     qrels[str(row['query-id'])] = {str(row['corpus-id']): row['score']}
        
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
    logging.info("Model: {}, lr: {:.2e}, batch_size: {}, epochs: {}".format(cfg.model.init.doc_model, cfg.training.lr, cfg.training.batch_size, cfg.training.max_epoch))
    logging.info("Normalize: {}, specialized mode: {}, pooling mode: {}".format(cfg.model.init.normalize, cfg.model.init.specialized_mode, cfg.model.init.aggregation_mode))
    loss_fn = MultipleRankingLoss(device=cfg.model.init.device)

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
        val_loss, average_precision, average_recall = validate(val_data, model, loss_fn, batch_size, epoch + 1, cfg.model.init.device)
        logging.info("VAL EPOCH: {:3d}, Average Loss: {:.5e}, Average Precision {}, Average Recall {}".format(epoch + 1, val_loss, round(average_precision*100,2), round(average_recall*100,2)))
        
        if val_loss < best_val_loss:
            logging.info(f'Found new best model on epoch: {epoch + 1}, new best validation loss {val_loss}')
            best_val_loss = val_loss
            logging.info(f'saving model checkpoint at epoch {epoch + 1}')
            save(model.state_dict(), f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}.pt')


if __name__ == '__main__':
    main()
