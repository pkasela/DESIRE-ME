import torch
from torch import clamp as t_clamp
from torch import nn
from torch import tensor
from torch import sum as t_sum
from torch import max as t_max
from torch import softmax
from torch import einsum
from torch.nn import functional as F


class BiEncoder(nn.Module):
    def __init__(self, doc_model, tokenizer, num_classes, device, mode='mean'):
        super(BiEncoder, self).__init__()
        # self.query_model = query_model.to(device)
        self.doc_model = doc_model.to(device)
        self.hidden_size = self.doc_model.config.hidden_size
        self.tokenizer = tokenizer
        self.device = device
        assert mode in ['max', 'mean'], 'Only max and mean pooling allowed'
        self.pooling = self.mean_pooling if mode == 'mean' else self.max_pooling
        
        self.num_classes = num_classes
        self.cls = nn.Linear(self.hidden_size, self.num_classes).to(device)
        
        self.query_embedding_changer = []
        for i in range(num_classes):
            self.query_embedding_changer.append(nn.Linear(self.hidden_size, self.hidden_size).to(device))
        
    def query_encoder(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)
        embeddings = self.doc_model(**encoded_input)
        return F.normalize(self.pooling(embeddings, encoded_input['attention_mask']), dim=-1)

    def doc_encoder(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)
        embeddings = self.doc_model(**encoded_input)
        return F.normalize(self.pooling(embeddings, encoded_input['attention_mask']), dim=-1)
    
    def forward(self, triplet_texts):
        query_embedding = self.query_encoder(triplet_texts[0])
        query_class = self.cls(query_embedding)
        
        pos_embedding = self.doc_encoder(triplet_texts[1])
        neg_embedding = self.doc_encoder(triplet_texts[2])
        
        return query_class, query_embedding, pos_embedding, neg_embedding

    def forward_random_neg(self, triplet):
        query_embedding = self.query_encoder(triplet[0])
        query_class = self.cls(query_embedding)
        
        query_embedding = self.query_embedder(query_embedding, query_class)
        
        pos_embedding = self.doc_encoder(triplet[1])
        if triplet[2][0] >= 0:
            neg_embedding = pos_embedding[tensor(triplet[2])]# self.doc_encoder(triplet_texts[2])
        else:
            print('A problem with batch size')
            neg_embedding = self.doc_encoder(['SEP'])

        return query_class, query_embedding, pos_embedding, neg_embedding
        
    def query_embedder(self, query_embedding, query_class):
        query_embs = [self.query_embedding_changer[i](query_embedding) for i in range(self.num_classes)]
        
        query_embs = torch.stack(query_embs, dim=1)
        
        query_class = F.softmax(query_class, dim=1)
        query_embs = einsum('bmd,bm->bd', query_embs, query_class)
        
        return F.normalize(query_embs)
        
    
    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return t_sum(token_embeddings * input_mask_expanded, 1) / t_clamp(input_mask_expanded.sum(1), min=1e-9)

    @staticmethod
    def max_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        return t_max(token_embeddings, 1)[0]
