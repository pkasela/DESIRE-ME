from ranx import optimize_fusion, Run, Qrels, fuse, compare
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    dev_run = Run.from_file(f'{cfg.testing.data_dir}/{cfg.model.init.save_model}_rerank_specialized_dev.json')
    dev_bm25_run = Run.from_file(cfg.testing.dev_bm25_run_path)
    
    dev_qrels = Qrels.from_file(cfg.testing.dev_qrels_path)
    
    best_params = optimize_fusion(
        qrels=dev_qrels,
        runs=[dev_bm25_run, dev_run],
        norm="min-max",
        method="wsum",
        metric="ndcg@10",  # The metric to maximize during optimization
        return_optimization_report=True
    )
    print(best_params)
    
    test_run = Run.from_file(f'{cfg.testing.data_dir}/{cfg.model.init.save_model}_rerank_specialized.json')
    test_bm25_run = Run.from_file(cfg.testing.bm25_run_path)
    
    test_qrels = Qrels.from_file(cfg.testing.qrels_path)
    
    combined_test_run = fuse(
        runs=[test_bm25_run, test_run],  
        norm="min-max",       
        method="wsum",        
        params=best_params[0],
    )
    
    models = [
        test_bm25_run,
        test_run,
        combined_test_run
    ]
    
    report = compare(
        qrels=test_qrels,
        runs=models,
        metrics=['precision@1', 'ndcg@3', 'ndcg@10', 'recall@100', 'map@100'],
        max_p=0.01
    )
    
    print(report)
    
    
if __name__ == "__main__":
    main()