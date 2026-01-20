
import torch
from torch import nn 
from helping_blocks import Multi_head_attention , Layer_norm , PositionalEncoding , Feed_forward
from preprocessing import english_sequences , eng_vocab_size ,eng_max_length

device = 'cuda' if torch.cuda.is_available() else 'cpu'


PAD_IDX = 0

class Encoder_layer(nn.Module):
    
    def __init__(self ,head_num , embed_dim ,ff_dim ,dropout=0.1):
        super().__init__()
        
        self.embed_dim =embed_dim
        self.attention =Multi_head_attention(head_num ,embed_dim)
        self.norm1 =Layer_norm(embed_dim)
        self.ff =Feed_forward(embed_dim ,ff_dim)
        
        self.norm2 =Layer_norm(embed_dim)
        self.dropout =nn.Dropout(dropout)
        
    def forward(self , x ,pad_mask):
        
       
        attention_out  =self.attention(x ,pad_mask)
        first_norm =self.norm1( x +self.dropout(attention_out))
        
        feed_forward =self.ff(first_norm)
        second_norm = self.norm2(first_norm + self.dropout(feed_forward))
        
        return second_norm





class Transformer_encoder(nn.Module):
    def __init__(self, vocab_size ,head_num, embed_dim, ff_dim,max_length ):
        super().__init__()
        
        
        self.embed_dim = embed_dim
        self.embedding =nn.Embedding(vocab_size ,embed_dim ,padding_idx=PAD_IDX)
        self.pos =PositionalEncoding(max_length ,embed_dim)
        self.encoder_layer =Encoder_layer(head_num ,embed_dim,ff_dim,dropout=0.1)
        
    def forward(self,x ):
        
        
        PAD_IDX = 0
        
        pad_mask = (x  == PAD_IDX)
        pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)
        x = x.long()
        
        embeddings =self.embedding(x)
        
        x = embeddings + self.pos(embeddings)
        
      
        
        encoder_out = self.encoder_layer( x, pad_mask)
        
        return encoder_out
        
        
# x =torch.rand(32,1500).to(device)
# torch.tensor(x)
# print(x.shape)



# model = Transformer_encoder(vocab_size=eng_vocab_size , head_num=8 ,embed_dim=128 ,ff_dim=512,max_length=1500).to(device)

# pre =model( x , english_sequences).to(device)

# print(pre.shape)