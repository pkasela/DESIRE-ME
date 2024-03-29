import click
import json
import pandas as pd
import os, pathlib
from beir import util

def iter_corpus_file(filename_corpus, dataset):
    with open(filename_corpus, 'rt') as corpusfile:
        for l in corpusfile:
            doc = dict()
            json_file = json.loads(l)
            doc['id'] = json_file["_id"]
            if dataset in ['nq', 'nq-train']:
                doc['contents'] = json_file["title"]+ '. ' + json_file["text"] # for the nq
            else:
                doc['contents'] = json_file["text"] # for hotpotqa
            doc['title'] = json_file["title"]
            yield doc
            
             
def iter_queries_file(filename_queries):
    with open(filename_queries, 'rt') as queries_file:
        for l in queries_file:
            doc = dict()
            json_file = json.loads(l)
            doc['id'] = json_file["_id"]
            doc['text'] = json_file["text"]
            yield doc

@click.command()
@click.option(
    "--data_folder",
    type=str,
    required=True,
)
@click.option(
    "--dataset",
    type=str,
    required=True,
)
def main(data_folder, dataset):
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "")
    data_path = util.download_and_unzip(url, out_dir)
    os.remove(os.path.join(pathlib.Path(__file__).parent.absolute(), f"{dataset}.zip"))

    filename_corpus = f'{data_folder}/corpus.jsonl'
    docs = [x for x in iter_corpus_file(filename_corpus, dataset)]

    os.makedirs(f'{data_folder}_serini_jsonl', exist_ok=True)
    with open(f'{data_folder}_serini_jsonl/corpus.jsonl', 'w') as outfile:
        for entry in docs:
            json.dump(entry, outfile)
            outfile.write('\n')
            
    filename_queries = f'{data_folder}/queries.jsonl'
    queries = [x for x in iter_queries_file(filename_queries)]


    qrel_df = pd.read_csv(f'{data_folder}/qrels/test.tsv', sep='\t')

    queries_df = pd.DataFrame(queries)
    queries_df.to_csv(f'{data_folder}/all_queries.tsv', sep='\t', header=False, index=None) 

    test_queries_df = queries_df[queries_df['id'].isin(qrel_df['query-id'].astype(str))]
    test_queries_df.to_csv(f'{data_folder}/queries.tsv', sep='\t', header=False, index=None) 

    if data_folder != 'nq' and data_folder != 'climate-fever':
        train_qrels = pd.read_csv(f'{data_folder}/qrels/train.tsv', sep='\t')
        dev_qrels = pd.read_csv(f'{data_folder}/qrels/dev.tsv', sep='\t')
        test_qrels = pd.read_csv(f'{data_folder}/qrels/test.tsv', sep='\t')
        
        dev_queries_df = queries_df[queries_df['id'].isin(dev_qrels['query-id'].astype(str))]
        dev_queries_df.to_csv(f'{data_folder}/dev_queries.tsv', sep='\t', header=False, index=None) 

        
        train_q_ids = set(train_qrels['query-id'].astype(str))
        dev_q_ids = set(dev_qrels['query-id'].astype(str))
        test_q_ids = set(test_qrels['query-id'].astype(str))
        
        train_queries = [q for q in queries if q['id'] in train_q_ids]
        dev_queries = [q for q in queries if q['id'] in dev_q_ids]
        test_queries = [q for q in queries if q['id'] in test_q_ids]

        with open(filename_queries.replace('queries.jsonl', 'test_queries.jsonl'), 'w') as f:
            for q in test_queries:
                q['_id'] = q['id']
                _ = q.pop('id')
                json.dump(q, f)
                f.write('\n')
        
        with open(filename_queries.replace('queries.jsonl', 'dev_queries.jsonl'), 'w') as f:
            for q in dev_queries:
                q['_id'] = q['id']
                _ = q.pop('id')
                json.dump(q, f)
                f.write('\n')
                                        
        with open(filename_queries.replace('queries.jsonl', 'train_queries.jsonl'), 'w') as f:
            for q in train_queries:
                q['_id'] = q['id']
                _ = q.pop('id')
                json.dump(q, f)
                f.write('\n')

if __name__ == '__main__':
    main()