import hydra
import logging
import os

from ranx import Qrels, Run, compare
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logging_file = "significance_test.log"
    logging.basicConfig(
        filename=os.path.join(cfg.dataset.logs_dir, logging_file),
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )
    
    logging.info('Loading BM25 file')
    bm25_run = Run.from_file(cfg.testing.bm25_run_path)
    bm25_run.name = 'BM25'

    try:
        logging.info('Loading COCO-DR-base_zeros file')
        cocodr_base_zeros = Run.from_file(f'{cfg.dataset.runs_dir}/cocodr-base-msmarco_zeros.lz4')
        cocodr_base_zeros.name = 'COCO-DR-base (Base)'
        logging.info('Loading COCO-DR-base_rand file')
        cocodr_base_rand = Run.from_file(f'{cfg.dataset.runs_dir}/cocodr-base-msmarco_rand.lz4')
        cocodr_base_rand.name = 'COCO-DR-base (rand)'
        logging.info('Loading COCO-DR-base_weight file')
        cocodr_base_weight = Run.from_file(f'{cfg.dataset.runs_dir}/cocodr-base-msmarco_desireme.lz4')
        cocodr_base_weight.name = 'COCO-DR-base (DESIRE-ME)'
        cocodr_base_exists = True
        try:
            logging.info('Loading COCO-DR-base finetined file')
            cocodr_base_finetined = Run.from_file(f'{cfg.dataset.runs_dir}/cocodr-base-msmarco_biencoder.lz4')
            cocodr_base_finetined.name = 'COCO-DR-base (fine tuned)'
            cocodr_base_finetined_exists = True
        except FileNotFoundError:
            cocodr_base_finetined_exists = False
    except FileNotFoundError:
        cocodr_base_exists = False

    try:
        logging.info('Loading contriever_zeros file')
        contriever_zeros = Run.from_file(f'{cfg.dataset.runs_dir}/contriever_zeros.lz4')
        contriever_zeros.name = 'Contriever (Base)'
        logging.info('Loading contriever_rand file')
        contriever_rand = Run.from_file(f'{cfg.dataset.runs_dir}/contriever_rand.lz4')
        contriever_rand.name = 'Contriever (rand)'
        logging.info('Loading contriever_weight file')
        contriever_weight = Run.from_file(f'{cfg.dataset.runs_dir}/contriever_desireme.lz4')
        contriever_weight.name = 'Contriever (DESIRE-ME)'
        contriever_exists = True
        try:
            logging.info('Loading Contriever Finetuned file')
            contriever_finetined = Run.from_file(f'{cfg.dataset.runs_dir}/contriever_biencoder.lz4')
            contriever_finetined.name = 'Contriever (fine tuned)'
            contriever_finetined_exists = True
        except FileNotFoundError:
            contriever_finetined_exists = False
    except FileNotFoundError:
        contriever_exists = False

    try:
        logging.info('Loading COCO-DR-large_zeros file')
        cocodr_large_zeros = Run.from_file(f'{cfg.dataset.runs_dir}/cocodr-large-msmarco_zeros.lz4')
        cocodr_large_zeros.name = 'COCO-DR-large (Base)'
        logging.info('Loading COCO-DR-large_rand file')
        cocodr_large_rand = Run.from_file(f'{cfg.dataset.runs_dir}/cocodr-large-msmarco_rand.lz4')
        cocodr_large_rand.name = 'COCO-DR-large (rand)'
        logging.info('Loading COCO-DR-large_weight file')
        cocodr_large_weight = Run.from_file(f'{cfg.dataset.runs_dir}/cocodr-large-msmarco_desireme.lz4')
        cocodr_large_weight.name = 'COCO-DR-large (DESIRE-ME)'
        cocodr_large_exists = True
        try:
            logging.info('Loading COCO-DR-large finetuned file')
            cocodr_large_finetined = Run.from_file(f'{cfg.dataset.runs_dir}/cocodr-large-msmarco_biencoder.lz4')
            cocodr_large_finetined.name = 'COCO-DR-large (fine tuned)'
            cocodr_large_finetined_exists = True
        except FileNotFoundError:
            cocodr_large_finetined_exists = False
    except:
        cocodr_large_exists = False

    logging.info('Loading qrels file')
    qrels = Qrels.from_file(cfg.testing.qrels_path)
   
    models = [bm25_run]
    tot_tests = 0
    if cocodr_base_exists:
        if cocodr_base_finetined_exists:
            models.extend([
                cocodr_base_zeros,
                cocodr_base_finetined,
                cocodr_base_rand,
                cocodr_base_weight
            ])
            tot_tests += 3
        else:
            models.extend([
                cocodr_base_zeros,
                cocodr_base_rand,
                cocodr_base_weight
            ])
            tot_tests += 2

    if contriever_exists:
        if contriever_finetined_exists:
            models.extend([
                contriever_zeros,
                contriever_finetined,
                contriever_rand,
                contriever_weight
            ])
            tot_tests += 3
        else:
            models.extend([
                contriever_zeros,
                contriever_rand,
                contriever_weight
            ])
            tot_tests += 2

    if cocodr_large_exists:
        if cocodr_large_finetined_exists:
            models.extend([
                cocodr_large_zeros,
                cocodr_large_finetined,
                cocodr_large_rand,
                cocodr_large_weight
            ])
            tot_tests += 3
        else:
            models.extend([
                cocodr_large_zeros,
                cocodr_large_rand,
                cocodr_large_weight
            ])
            tot_tests += 2
    
    evaluation_report = compare(
        qrels,
        models,
        ['map@100', 'mrr@10', 'recall@100', 'ndcg@10', 'precision@1', 'ndcg@3'],
        max_p=.01/tot_tests
    )
    
    print(evaluation_report)
    with open(os.path.join(cfg.dataset.logs_dir, 'latex_table.tex'), 'w') as f:
        f.write(evaluation_report.to_latex())
    logging.info(f'\n{evaluation_report}\n')

if __name__ == '__main__':
    main()