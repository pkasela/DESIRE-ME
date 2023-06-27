import csv
import json
import pandas as pd
import tqdm

from ranx import Run, Qrels, evaluate

data_folder = 'fever'

qrel_df = pd.read_csv(f'{data_folder}/qrels/test.tsv', sep='\t')
qrels = {}

for index, row in qrel_df.iterrows():
    q_id = str(row['query-id']) 
    
    if not q_id in qrels:
        qrels[q_id] = {}
    
    qrels[q_id][str(row['corpus-id'])] = row['score']

bm25_serini = pd.read_csv(f'{data_folder}/run.txt', sep=' ', header=None, quoting=csv.QUOTE_NONE)
    
run = {}
for _id, row in tqdm.tqdm(bm25_serini[[0, 2, 4]].iterrows()):
    q_id = str(row[0])
    if not q_id in run:
        run[q_id] = {}
        
    run[q_id][str(row[2])] = row[4]

with open(f'{data_folder}/qrels.json', 'w') as f:
    json.dump(qrels, f, indent=2)
    
with open(f'{data_folder}/bm25_run.json', 'w') as f:
    json.dump(run, f, indent=2)

ranx_qrels = Qrels(qrels)
ranx_run = Run(run)

print(evaluate(ranx_qrels, ranx_run, ['map@100', 'mrr@10', 'recall@100', 'precision@5', 'ndcg@10']))