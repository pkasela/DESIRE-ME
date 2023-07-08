from ranx import optimize_fusion, Run, Qrels, fuse, compare
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os

logger = logging.getLogger(__name__)



@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logging_file = "optimized.log"
    logging.basicConfig(
        filename=os.path.join(cfg.dataset.logs_dir, logging_file),
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )
    logging.info(f'Testing on: {cfg.testing.data_dir}')
    if 'nq' in cfg.testing.data_dir:
        best_ns_params = [{'weights': (0.5, 0.5)}]
        best_s_params = [{'weights': (0.5, 0.5)}]
    else:
        dev_s_run = Run.from_file(f'{cfg.testing.data_dir}/{cfg.model.init.save_model}_rerank_specialized_dev.json')
        dev_ns_run = Run.from_file(f'{cfg.testing.data_dir}/{cfg.model.init.save_model}_rerank_non_specialized_dev.json')
        dev_bm25_run = Run.from_file(cfg.testing.dev_bm25_run_path)
        
        dev_qrels = Qrels.from_file(cfg.testing.dev_qrels_path)
        
        best_s_params = optimize_fusion(
            qrels=dev_qrels,
            runs=[dev_bm25_run, dev_s_run],
            norm="min-max",
            method="wsum",
            metric="ndcg@10",  # The metric to maximize during optimization
            step=.1,
            return_optimization_report=True
        )
        logging.info('Best parameter for specialized model')
        logging.info(str(best_s_params[0]))
        logging.info(f'\n{str(best_s_params[1])}')
        
        
        best_ns_params = optimize_fusion(
            qrels=dev_qrels,
            runs=[dev_bm25_run, dev_ns_run],
            norm="min-max",
            method="wsum",
            metric="ndcg@10",  # The metric to maximize during optimization
            step=.1,
            return_optimization_report=True
        )
        logging.info('Best parameter for non specialized model')
        logging.info(str(best_s_params[0]))
        logging.info(f'\n{str(best_s_params[1])}')
        
    test_s_run = Run.from_file(f'{cfg.testing.data_dir}/{cfg.model.init.save_model}_rerank_specialized.json')
    test_s_run.name = 'S'
    test_ns_run = Run.from_file(f'{cfg.testing.data_dir}/{cfg.model.init.save_model}_rerank_non_specialized.json')
    test_ns_run.name = 'NS'
    test_bm25_run = Run.from_file(cfg.testing.bm25_run_path)
    test_bm25_run.name = 'BM25'
    
    test_qrels = Qrels.from_file(cfg.testing.qrels_path)
    
    combined_test_s_run = fuse(
        runs=[test_bm25_run, test_s_run],  
        norm="min-max",       
        method="wsum",        
        params=best_s_params[0],
    )
    combined_test_s_run.name = 'BM25 + S'
    
    combined_test_ns_run = fuse(
        runs=[test_bm25_run, test_ns_run],  
        norm="min-max",       
        method="wsum",        
        params=best_ns_params[0],
    )
    combined_test_ns_run.name = 'BM25 + NS'
    
    models = [
        test_bm25_run,
        test_ns_run,
        test_s_run,
        combined_test_ns_run,
        combined_test_s_run
    ]
    
    report = compare(
        qrels=test_qrels,
        runs=models,
        metrics=['map@100', 'mrr@10', 'recall@100', 'precision@5', 'ndcg@10', 'precision@1', 'ndcg@3'],
        max_p=0.01
    )
    
    print(report)
    logging.info(f'\n{report}')
    
    
if __name__ == "__main__":
    main()