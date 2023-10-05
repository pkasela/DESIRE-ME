import json
from indxr import Indxr
import pandas as pd
import tqdm
import numpy as np

data_folder = '../nq-train'

if 'nq-train' in data_folder:
    queries = Indxr(f'{data_folder}/queries.jsonl', key_id='_id')
else:
    queries = Indxr(f'{data_folder}/train_queries.jsonl', key_id='_id')

corpus = Indxr(f'{data_folder}/wiki_corpus.jsonl', key_id='_id')

qrel_df = pd.read_csv(f'{data_folder}/qrels/train.tsv', sep='\t')
qrels = {}

        
for index, row in qrel_df.iterrows():
    q_id = str(row['query-id']) 
    
    if not q_id in qrels:
        qrels[q_id] = {}
    
    qrels[q_id][str(row['corpus-id'])] = row['score']


found_cat = 0
how_many = []
category_frequency = []
for q in tqdm.tqdm(queries):
    rel_docs = qrels[q['_id']]
    categories = []
    for doc_id in rel_docs:
        categories.extend(corpus.get(doc_id)['category'])
        categories = list(set(categories))
        for c in categories:
            category_frequency.append(c)
    how_many.append(len(categories))
    if categories:
        found_cat += 1

print(found_cat / len(queries))
print(np.mean(how_many))
print(np.std(how_many))

print(pd.Series(category_frequency).value_counts())