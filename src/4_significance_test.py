import hydra
import logging

from ranx import Qrels, Run, compare
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # logging_file = "significance_test.log"
    logging.basicConfig(
        # filename=os.path.join(cfg.dataset.logs_dir, logging_file),
        # filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )

    logging.info('Loading BM25 file')
    bm25_run = Run.from_file(cfg.testing.bm25_run_path)
    bm25_run.name = 'BM25'

    logging.info('Loading COCO-DR-base_zeros file')
    cocodr_base_zeros = Run.from_file(f'{cfg.dataset.runs_dir}/cocodr-base-msmarco_fullrank_zeros.json')
    cocodr_base_zeros.name = 'COCO-DR-base (zeros)'
    logging.info('Loading COCO-DR-base_rand file')
    cocodr_base_rand = Run.from_file(f'{cfg.dataset.runs_dir}/cocodr-base-msmarco_fullrank_rand.json')
    cocodr_base_rand.name = 'COCO-DR-base (rand)'
    logging.info('Loading COCO-DR-base_weight file')
    cocodr_base_weight = Run.from_file(f'{cfg.dataset.runs_dir}/cocodr-base-msmarco_fullrank_weight.json')
    cocodr_base_weight.name = 'COCO-DR-base (weight)'
    
    logging.info('Loading contriever_zeros file')
    contriever_zeros = Run.from_file(f'{cfg.dataset.runs_dir}/contriever_fullrank_zeros.json')
    contriever_zeros.name = 'Contriever (zeros)'
    logging.info('Loading contriever_rand file')
    contriever_rand = Run.from_file(f'{cfg.dataset.runs_dir}/contriever_fullrank_rand.json')
    contriever_rand.name = 'Contriever (rand)'
    logging.info('Loading contriever_weight file')
    contriever_weight = Run.from_file(f'{cfg.dataset.runs_dir}/contriever_fullrank_weight.json')
    contriever_weight.name = 'Contriever (weight)'

    logging.info('Loading COCO-DR-large_zeros file')
    cocodr_large_zeros = Run.from_file(f'{cfg.dataset.runs_dir}/cocodr-large-msmarco_fullrank_zeros.json')
    cocodr_large_zeros.name = 'COCO-DR-large (zeros)'
    logging.info('Loading COCO-DR-large_rand file')
    cocodr_large_rand = Run.from_file(f'{cfg.dataset.runs_dir}/cocodr-large-msmarco_fullrank_rand.json')
    cocodr_large_rand.name = 'COCO-DR-large (rand)'
    logging.info('Loading COCO-DR-large_weight file')
    cocodr_large_weight = Run.from_file(f'{cfg.dataset.runs_dir}/cocodr-large-msmarco_fullrank_weight.json')
    cocodr_large_weight.name = 'COCO-DR-large (weight)'

    logging.info('Loading qrels file')
    qrels = Qrels.from_file(cfg.testing.qrels_path)
    
    models = [
        bm25_run,
        cocodr_base_zeros,
        cocodr_base_rand,
        cocodr_base_weight,
        contriever_zeros,
        contriever_rand,
        contriever_weight,
        cocodr_large_zeros,
        cocodr_large_rand,
        cocodr_large_weight,
    ]

    evaluation_report = compare(
        qrels, 
        models, 
        ['map@100', 'mrr@10', 'recall@100', 'ndcg@10', 'precision@1', 'ndcg@3'],
        max_p=.01/4
    )
    
    print(evaluation_report)

if __name__ == '__main__':
    main()