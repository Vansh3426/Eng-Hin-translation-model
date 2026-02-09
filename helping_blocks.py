import torch 
from torch import nn
import math


class Multi_head_attention(nn.Module):
    
    def __init__(self , head_num,embed_dim):
        
        super().__init__()
        
        self.embed_dim = embed_dim
        self.head_num =head_num
        self.head_dim = embed_dim // head_num
        
        assert embed_dim % head_num ==0
        
        self.q =nn.Linear(embed_dim,embed_dim,bias=False)
        self.k =nn.Linear(embed_dim,embed_dim,bias=False)
        self.v =nn.Linear(embed_dim,embed_dim,bias=False)
        
        self.out =nn.Linear(embed_dim ,embed_dim)
        
    def forward(self,x ,pad_mask= None):
        
        # print(x.shape)
        B ,T ,E = x.shape
        
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        
        
        Q = Q.view(B ,T ,self.head_num ,self.head_dim).transpose(1,2)
        K = K.view(B ,T ,self.head_num ,self.head_dim).transpose(1,2)
        V = V.view(B ,T ,self.head_num ,self.head_dim).transpose(1,2)
        
        
        # print(f'Q  :  {Q.shape}')
        # print(f'Q  :  {Q}')
        # print(f'k  :  {K.shape}')
        # print(f'V  :  {V.shape}')
        
        scores =Q  @  K.transpose(-2,-1)
        
        # print(f"scores after dot product  : {scores.shape}")
        
        scores = scores / math.sqrt(self.head_dim)
        # print(f' scores before padmask  : {scores.shape}')
        if pad_mask is not None:
             # [B, 1, 1, T]
            
            # print(pad_mask_expanded.shape)
            scores = scores.masked_fill(pad_mask, -1e9)
            # print(f' scores after padmask  : {scores.shape}')
        # print(f'scores after scaling  :  {scores}')
        
        prob= torch.softmax(scores, dim=-1)
        
        # print(f'prob  :  {prob.shape}')
        
        context = prob @  V 
        
        context =context.transpose(1,2).contiguous()
        context = context.view(B , T , E)
        context =self.out(context)
        # print(f' contex ::  {context.shape}')
        
        return context 
    


# x(batch ,seq , embedding)

# x =torch.rand(1,5,8)

# print(x.shape)

# atten =Multi_head_attention(head_num=2 ,embed_dim=8)

# ouput =atten(x)

# print(ouput)



# class positional_encoding(nn.Module):
    
#     def __init__(self,max_length ,model_dim):
        
#         self.max_l = max_length
        
#         self.d_model = model_dim
        
#     def forward(self):
        
#         i = torch.arange(0 , self.d_model ,2 ).float()
        
#         denominator =torch.pow(10000 , (i/self.d_model))
        
#         pos = torch.arange(self.max_l).reshape(self.max_l ,1)
        
#         even_pos = torch.sin(pos/denominator)
#         odd_pos = torch.cos(pos/denominator)
        
#         stacked = torch.stack([even_pos , odd_pos],dim=2)
#         pos_all = torch.flatten(stacked ,start_dim=1 ,end_dim=2)
        
#         return pos_all
        


class PositionalEncoding(nn.Module):
    def __init__(self, max_length, d_model):
        super().__init__()
        self.max_length = max_length
        self.d_model = d_model

        # Create the positional encoding matrix once, register as buffer
        pe = torch.zeros(max_length, d_model)
        # print(f'first pe shape {pe.shape}')
        
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        # print(f'postion shape  {position.shape}')
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # print(div_term.shape)
        
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        # print(f'even pe shape :{pe.shape}')
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        # print(f' odd pe shape :{pe.shape}')

        pe = pe.unsqueeze(0)
        # print(f' pe shape :{pe.shape}')# shape: (1, max_len, d_model) for broadcasting
        self.register_buffer('pe', pe)  # not a parameter, but saved with the model

    def forward(self, x):
       
        # print(x.shape)
        seq_len = x.size(1)
        # print(x.shape)
        
        x = self.pe[:, :seq_len, :].to(x.device)
        
        # Add positional encoding to input embeddings
        return x

        
# x =torch.rand(32,6,128)

# model = PositionalEncoding(35 , 128)

# pred = model(x)

# print(pred.shape)        
        
# pos = PositionalEncoding(10,6)

# ouput = pos.forward()

# print(ouput)



class Layer_norm(nn.Module):
    
    def __init__(self , embedding_dim , eps =1e-5):
        
        super().__init__()
        
        self.gamma =nn.Parameter(torch.ones(embedding_dim))
        self.beta =nn.Parameter(torch.zeros(embedding_dim))
        self.eps =eps
        
    def forward(self , x):
        
        # x.shape = batch_size , seq_length , embedding_dim 
        
        
        mean = x.mean(dim=-1 , keepdim =True)
        var = x.var(dim=-1 ,keepdim=True ,unbiased=True)
        
        x_hat = (x - mean)/torch.sqrt(var + self.eps)
        
        x_norm = self.gamma * x_hat + self.beta
        
        return x_norm
    

class Feed_forward(nn.Module):
     def __init__(self ,embedding_dim ,ff_dim ):
         super().__init__()
         
         self.layer1= nn.Linear(embedding_dim ,ff_dim)
         self.layer2 = nn.Linear(ff_dim,embedding_dim)
         self.relu = nn.ReLU()
         
     def forward(self , x):
         
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        
        return x
    
    


class Masked_multi_head_attention(nn.Module):
    
    def __init__(self , head_num,embed_dim):
        
        super().__init__()
        
        self.embed_dim = embed_dim
        self.head_num =head_num
        self.head_dim = embed_dim // head_num
        
        assert embed_dim % head_num ==0
        
        self.q =nn.Linear(embed_dim,embed_dim,bias=False)
        self.k =nn.Linear(embed_dim,embed_dim,bias=False)
        self.v =nn.Linear(embed_dim,embed_dim,bias=False)
        
        self.out =nn.Linear(embed_dim ,embed_dim)
        
    def forward(self,x ,pad_mask= None):
        
        B ,T ,E = x.shape
        
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        
        
        Q = Q.view(B ,T ,self.head_num ,self.head_dim).transpose(1,2)
        K = K.view(B ,T ,self.head_num ,self.head_dim).transpose(1,2)
        V = V.view(B ,T ,self.head_num ,self.head_dim).transpose(1,2)
        
        
        # print(f'Q  :  {Q.shape}')
        # print(f'Q  :  {Q}')
        # print(f'k  :  {K.shape}')
        # print(f'V  :  {V.shape}')
        
        scores =Q  @  K.transpose(-2,-1)
        
        # print(f"scores : {scores}")
        
        scores = scores / math.sqrt(self.head_dim)
        
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  
        # print(causal_mask.shape)
        if pad_mask is not None:
            # print(causal_mask.shape)
            combined_mask = causal_mask & (~pad_mask)
        else:
            combined_mask = causal_mask

   
        scores = scores.masked_fill(~combined_mask, -1e9)

        # print(f'scores after scaling  :  {scores}')
        
        query_x_key= torch.softmax(scores, dim=-1)
        
        # print(f'prob  :  {prob.shape}')
        
        context = query_x_key @  V 
        
        context =context.transpose(1,2).contiguous()
        context = context.view(B , T , E)
        
        # print(f' contex ::  {context.shape}')
        context = self.out(context)
        return context 
    
    
    

class Cross_attention(nn.Module):
    
    def __init__(self , head_num,embed_dim):
        
        super().__init__()
        
        self.embed_dim = embed_dim
        self.head_num =head_num
        self.head_dim = embed_dim // head_num
        
        assert embed_dim % head_num ==0
        
        self.q =nn.Linear(embed_dim,embed_dim,bias=False)
        self.k =nn.Linear(embed_dim,embed_dim,bias=False)
        self.v =nn.Linear(embed_dim,embed_dim,bias=False)
        
        self.out =nn.Linear(embed_dim ,embed_dim)
        
    def forward(self, encoder_output, masked_atten_output):
        
        B, tgt_len, E = masked_atten_output.shape
        _, src_len, _ = encoder_output.shape

        
        Q = self.q(masked_atten_output)
        K = self.k(encoder_output)
        V = self.v(encoder_output)
        
        
        Q = Q.view(B , tgt_len ,self.head_num ,self.head_dim).transpose(1,2)
        K = K.view(B ,src_len ,self.head_num ,self.head_dim).transpose(1,2)
        V = V.view(B ,src_len ,self.head_num ,self.head_dim).transpose(1,2)
        
        
        # print(f'Q  :  {Q.shape}')
        # print(f'Q  :  {Q}')
        # print(f'k  :  {K.shape}')
        # print(f'V  :  {V.shape}')
        
        scores =Q  @  K.transpose(-2,-1)
        
        # print(f"scores after dot product  : {scores.shape}")
        
        scores = scores / math.sqrt(self.head_dim)
        # print(f' scores before padmask  : {scores.shape}')
  
        # print(f'scores after scaling  :  {scores}')
        
        prob= torch.softmax(scores, dim=-1)
        
        # print(f'prob  :  {prob.shape}')
        
        context = prob @  V 
        
        context =context.transpose(1,2).contiguous()
        context = context.view(B , tgt_len , E)
        context =self.out(context)
        # print(f' contex ::  {context.shape}')
        
        return context 
    