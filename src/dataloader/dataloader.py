import torch
from indxr import Indxr
import random


class LoadTrainNQData(torch.utils.data.Dataset):
    def __init__(self, query_path, corpus_path, qrels, category_to_label):
        self.query_path  = query_path
        self.corpus_path = corpus_path
        self.qrels = qrels
        self.cat_to_label = category_to_label
        
        self.init_query()
        self.init_corpus()
        
    def init_query(self):
        self.queries = Indxr(self.query_path, key_id='_id')
        
    def init_corpus(self):
        self.corpus = Indxr(self.corpus_path, key_id='_id')
        
    def __getitem__(self, idx):
        query = self.queries[idx]
        query_text = query['text'].lower()
        
        pos_ids = set(self.qrels[query['_id']])
        pos_id = str(random.choice(list(pos_ids)))
        pos_doc = self.corpus.get(pos_id)
        
        pos_category = pos_doc.get('category', [])
        category_tensor = torch.zeros(len(self.cat_to_label))
        
        for cat in pos_category:
            category_tensor[self.cat_to_label[cat]] = 1
        return {
            'question': query_text,
            'pos_text': pos_doc.get('title', '').lower() + ' ' + pos_doc['text'].lower(),
            'pos_category': category_tensor 
        }
        
    def __len__(self):
        return len(self.queries)
        

def in_batch_negative_collate_fn(batch):
    question_texts = [x['query_text'] for x in batch]
    
    pos_texts = list(enumerate(x['pos_text'] for x in batch))
    if len(pos_texts) > 1:
        neg_texts = [random.choice(pos_texts[:i] + pos_texts[i+1:])[0] for i in range(len(pos_texts))]
    else: 
        neg_texts = [-1]
    
    return {
        'question': question_texts,
        'pos_text': [x.get('pos_text') for x in batch],
        'pos_category': [x.get('pos_category') for x in batch],
    }
