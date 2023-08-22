
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_


class TextEncoder(torch.nn.Module):
    def __init__(self,
                 bert_model,
                 item_embedding_dim,
                 word_embedding_dim,
                 args):
        super(TextEncoder, self).__init__()
        self.bert_model = bert_model
        self.text_pooler = nn.Linear(word_embedding_dim, item_embedding_dim)

        xavier_normal_(self.text_pooler.weight.data)
        if self.text_pooler.bias is not None:
            constant_(self.text_pooler.bias.data, 0)

        self.activate = nn.ReLU() 

    def forward(self, text):
        batch_size, num_words = text.shape
        num_words = num_words // 2
        text_ids = torch.narrow(text, 1, 0, num_words)
        text_attmask = torch.narrow(text, 1, num_words, num_words)

        hidden_states = self.bert_model(input_ids=text_ids, attention_mask=text_attmask)[0] 
        hidden_states = self.text_pooler(hidden_states)
        return hidden_states


class TextEmbedding(torch.nn.Module):
    def __init__(self, args, bert_model):
        super(TextEmbedding, self).__init__()
        self.args = args
        # we use the title of item with a fixed length.
        self.text_length = args.num_words_title * 2 # half for mask
    
        self.text_encoders = TextEncoder(bert_model, args.embedding_dim, args.word_embedding_dim ,args)

    def forward(self, news):
        text_vectors = self.text_encoders(torch.narrow(news, 1, 0, self.text_length))
        return text_vectors
