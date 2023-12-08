import torch as t
import torch.nn as nn


class Memorizer(nn.Module):
    def __init__(self, number_of_classes):
        super(Memorizer, self).__init__()
        self.embed = nn.Embedding(number_of_classes, 512)
        self.pos_enc = PositionalEncoding(512, 0.1)
        self.transformer = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6)
        self.linear = nn.Linear(512, number_of_classes)

    def forward(self, x, y):
        x = self.embed(x)
        x = self.pos_enc(x)
        y = self.embed(y)
        y = self.pos_enc(y)
        x = self.transformer(x, y)
        x = self.linear(x)
        
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        # Compute the positional encodings once in log space.
        pe = t.zeros(max_len, d_model)
        position = t.arange(0, max_len, dtype=t.float).unsqueeze(1)
        div_term = t.exp(t.arange(0, d_model, 2).float() * (-t.log(t.tensor(10000.0)) / d_model))
        pe[:, 0::2] = t.sin(position * div_term)
        pe[:, 1::2] = t.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * t.sqrt(t.tensor(self.d_model, dtype=t.float))
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

