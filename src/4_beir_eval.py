"""
Code provided in the official beir library at the following link: https://github.com/beir-cellar/beir
Just added the flag option for ease of use using the click library
"""
import hydra
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from omegaconf import DictConfig, OmegaConf
from ranx import Run, Qrels
from sentence_transformers.models import Pooling

import logging
import pathlib, os


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout
    
    dataset = cfg.testing.name
    model_name = cfg.model.doc_model
    score_function = 'cos_sim' if cfg.model.init.normalize else 'dot'
    pooling_function = cfg.model.init.aggregation_mode
    
    #### Download scifact.zip dataset and unzip the dataset
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)

    #### Provide the data_path where scifact has been downloaded and unzipped
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    l_model = models.SentenceBERT(model_name)
    if pooling_function == 'cls':
        logging.info('Using CLS function as pooling method')
        l_model.q_model._modules['1'] = Pooling(l_model.q_model.get_sentence_embedding_dimension(), 'cls')

    #### Load the SBERT model and retrieve using cosine-similarity
    model = DRES(l_model, batch_size=16)
    retriever = EvaluateRetrieval(model, score_function=score_function) # or "cos_sim" for cosine similarity
    results = retriever.retrieve(corpus, queries)

    #### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] 
    retriever.k_values = [1,3,5,10,100]
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

    qrels = Qrels(qrels)
    run_results = Run(results, name=model_name.replace("/","_"))
    run_results.save(f'{cfg.dataset.runs_dir}/{cfg.model.init.save_model}_zeros.lz4')

if __name__ == '__main__':
    main()