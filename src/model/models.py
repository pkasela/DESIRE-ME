import torch
from torch import clamp as t_clamp
from torch import nn
from torch import tensor
from torch import sum as t_sum
from torch import max as t_max
from torch import softmax, sigmoid
from torch import einsum
from torch.nn import functional as F


class QuerySpecializer(nn.Module):
    def __init__(self, hidden_size, device):
        super(QuerySpecializer, self).__init__()
        
        self.hidden_size = hidden_size
        self.device = device
        self.query_embedding_changer_1 = nn.Linear(self.hidden_size, self.hidden_size//2).to(device)
        self.query_embedding_changer_4 = nn.Linear(self.hidden_size//2, self.hidden_size).to(device)
        
    def forward(self, query_embs):
        query_embs = F.relu(self.query_embedding_changer_1(query_embs))
        query_embs = self.query_embedding_changer_4(query_embs)
        
        return query_embs


class SpecialziedBiEncoder(nn.Module):
    def __init__(
        self, 
        doc_model, 
        tokenizer, 
        num_classes, 
        normalize=False,
        specialized_mode='weight',
        pooling_mode='mean', 
        device='cpu',        
    ):
        super(SpecialziedBiEncoder, self).__init__()
        self.doc_model = doc_model.to(device)
        self.hidden_size = self.doc_model.config.hidden_size
        self.tokenizer = tokenizer
        self.device = device
        self.normalize = normalize
        assert specialized_mode in ['weight', 'zeros', 'ones', 'rand'], 'Only weight, zeros, ones and rand specialzed mode allowed'
        self.specialized_mode = specialized_mode
        assert pooling_mode in ['max', 'mean', 'cls', 'identity'], 'Only cls, identity, max and mean pooling allowed'
        if pooling_mode == 'mean':
            self.pooling = self.mean_pooling
        elif pooling_mode == 'max':
            self.pooling = self.max_pooling
        elif pooling_mode == 'cls':
            self.pooling = self.cls_pooling
        elif pooling_mode == 'identity':
            self.pooling = self.identity
        
        self.num_classes = num_classes
        self.init_cls()
        
        self.query_specializer = nn.ModuleList([QuerySpecializer(self.hidden_size, self.device) for _ in range(self.num_classes)])    
        
    def query_encoder(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)
        embeddings = self.doc_model(**encoded_input)
        if self.normalize:
            return F.normalize(self.pooling(embeddings, encoded_input['attention_mask']), dim=-1)
        return self.pooling(embeddings, encoded_input['attention_mask'])
        
    
    def init_cls(self):
        self.cls_1 = nn.Linear(self.hidden_size, self.hidden_size*2).to(self.device)
        self.cls_2 = nn.Linear(self.hidden_size*2, self.hidden_size*4).to(self.device)
        self.cls_3 = nn.Linear(self.hidden_size*4, self.num_classes).to(self.device)
        
    
    def query_encoder_with_context(self, sentences):
        query_embedding = self.query_encoder(sentences)
        query_class = self.cls(query_embedding)
        query_embedding = self.query_embedder(query_embedding, query_class)
        return query_embedding

    def query_encoder_with_context_val(self, sentences):
        query_embedding = self.query_encoder(sentences)
        query_class = self.cls(query_embedding)
        query_embedding = self.val_query_embedder(query_embedding, query_class)
        return query_embedding
        

    def doc_encoder(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)
        embeddings = self.doc_model(**encoded_input)
        if self.normalize:
            return F.normalize(self.pooling(embeddings, encoded_input['attention_mask']), dim=-1)
        return self.pooling(embeddings, encoded_input['attention_mask'])
        
    def cls(self, query_embedding):
        x1 = F.relu(self.cls_1(query_embedding))
        x2 = F.relu(self.cls_2(x1))
        out = self.cls_3(x2)
        
        return out
    
    def forward(self, data):
        with torch.no_grad():
            query_embedding = self.query_encoder(data[0])
        query_class = self.cls(query_embedding)
        
        query_embedding = self.query_embedder(query_embedding, query_class)
        
        with torch.no_grad():
            pos_embedding = self.doc_encoder(data[1])
        
        return query_class, query_embedding, pos_embedding

        
    def val_forward(self, data):
        with torch.no_grad():
            query_embedding = self.query_encoder(data[0])
            query_class = self.cls(query_embedding)
        

            query_embedding = self.query_embedder(query_embedding, query_class)
        
            pos_embedding = self.doc_encoder(data[1])
        
        return query_class, query_embedding, pos_embedding


    def query_embedder(self, query_embedding, query_class):
        query_embs = [self.query_specializer[i](query_embedding) for i in range(self.num_classes)]
        
        query_embs = torch.stack(query_embs, dim=1)
        
        if self.specialized_mode == 'weight':
            query_class = sigmoid(query_class.detach())
        if self.specialized_mode == 'zeros':
            query_class = torch.zeros(query_class.shape).to(self.device)
        if self.specialized_mode == 'ones':
            query_class = torch.ones(query_class.shape).to(self.device)
        if self.specialized_mode == 'rand':
            query_class = torch.rand(query_class.shape).to(self.device)
        
        query_embs = F.normalize(einsum('bmd,bm->bd', query_embs, query_class), dim=-1, eps=1e-6) + query_embedding

        if self.normalize:
            return F.normalize(query_embs, dim=-1)
        return query_embs
    
    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return t_sum(token_embeddings * input_mask_expanded, 1) / t_clamp(input_mask_expanded.sum(1), min=1e-9)

    @staticmethod
    def cls_pooling(model_output, attention_mask):
        last_hidden = model_output["last_hidden_state"]
        # last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden[:, 0]

    @staticmethod
    def identity(model_output, attention_mask):
        return model_output['pooler_output']
    
    @staticmethod
    def max_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        return t_max(token_embeddings, 1)[0]
