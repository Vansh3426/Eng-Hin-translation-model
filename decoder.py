import torch
from torch import nn
import math
from helping_blocks import Masked_multi_head_attention ,Multi_head_attention ,PositionalEncoding , Layer_norm ,Feed_forward ,Cross_attention
from preprocessing_02 import vocab_size ,max_length ,dataloader
from encoder import Transformer_encoder



device = 'cuda'  if torch.cuda.is_available else 'cpu'

torch.manual_seed(32)
torch.cuda.manual_seed(42)




class Decoder_layer(nn.Module):
    def __init__(self , embed_dim , head_num , ff_dim , dropout = 0.1):
        super().__init__()
        
        
        self.masked_atten = Masked_multi_head_attention(head_num ,embed_dim)
        self.cross_atten =Cross_attention(head_num ,embed_dim)
        self.first_norm = Layer_norm(embed_dim)
        self.second_norm =Layer_norm(embed_dim)
        self.third_norm =Layer_norm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ff =Feed_forward(embed_dim ,ff_dim)
        
        
    def forward(self , embed_y ,pad_mask, Encoder_ouput ):
        
       
        
        masked_atten_output= self.masked_atten(embed_y ,pad_mask)
        first_norm =self.first_norm( embed_y +self.dropout(masked_atten_output))
        
        cross_atten_context = self.cross_atten(Encoder_ouput , first_norm)
        
        second_norm =self.second_norm( first_norm + self.dropout(cross_atten_context))
        
        ff_layer = self.ff(second_norm)
        third_norm = self.third_norm( second_norm + self.dropout(ff_layer) )
        
        return third_norm
        
        
        

   
class Transformer_decoder(nn.Module):
    
    def __init__(self , vocab_size , head_num ,embed_dim ,ff_dim ,max_length ,):
        
        super().__init__()
        PAD_IDX = -1
        self.embed_dim = embed_dim
        self.embedding =nn.Embedding(vocab_size , embed_dim , padding_idx=PAD_IDX)
        self.pos = PositionalEncoding(max_length ,embed_dim)
        self.decoder =Decoder_layer(embed_dim ,head_num,ff_dim)
        
       
        
        
    def forward(self , y ,Encoder_output ):
        
        decoder_training = y
       
        
        
        
        PAD_IDX = -1
        pad_mask = (decoder_training == PAD_IDX)
        pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)
        
        embedding = self.embedding(decoder_training)
        embedding = embedding * math.sqrt(self.embed_dim)
        embed_y = embedding + self.pos(embedding) 
        
        
        
        decoder_output = self.decoder(embed_y , pad_mask , Encoder_output)      
        
        
        return decoder_output 
        
        
       


        
        

        
class Transformer_runner(nn.Module):
    
    def __init__(self , vocab_size , enc_head_num ,encoder_embed_dim ,encoder_ff_dim  ,dec_head_num , decoder_embed_dim ,decoder_ff_dim ,max_length ):
        
        super().__init__()
        
        
        self.encoder = Transformer_encoder(vocab_size , head_num=enc_head_num , embed_dim = encoder_embed_dim ,ff_dim=encoder_ff_dim ,max_length =max_length )
        self.decoder = Transformer_decoder(vocab_size , head_num=dec_head_num , embed_dim = decoder_embed_dim ,ff_dim=decoder_ff_dim ,max_length =max_length)
        self.linear =nn.Linear(decoder_embed_dim , vocab_size)
        
        
    def forward(self , x , y ):
       
        encoder_output = self.encoder( x )
        decoder_output  = self.decoder( y , encoder_output)   
        
        pred= self.linear(decoder_output)
        
        # print(pred.shape)
        # output = torch.softmax(pred , dim= -1)
        # output =pred
       
        return pred




model = Transformer_runner(vocab_size , enc_head_num = 4 ,encoder_embed_dim =128 ,
                           encoder_ff_dim =256 ,max_length = max_length ,dec_head_num =4 , decoder_embed_dim  = 128,
                           decoder_ff_dim = 256).to(device)




# model_args = {
#     "eng_vocab_size": eng_vocab_size,
#     "enc_head_num": 8,
#     "encoder_embed_dim": 128,
#     "encoder_ff_dim": 512,
#     "eng_max_length": 15,

#     "hin_vocab_size": hin_vocab_size,
#     "dec_head_num": 8,
#     "decoder_embed_dim": 128,
#     "decoder_ff_dim": 512,
#     "hin_max_length": 20
# }


PAD_IDX = -1

loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(params=model.parameters() ,lr = 0.0003)




if __name__ == "__main__":
    
    model.train()

    epochs =30

    for epoch in range(epochs):
        
        total_loss = 0
        total_batch = 0

        for batch in (dataloader):
        
            X ,y = batch['input_ids'].to(device) ,batch['target_ids'].to(device)
         
            decoder_input = y[: , :-1]
            decoder_loss = y[: , 1:]
            
            pred  = model(X ,decoder_input)
            
            pred = pred.reshape(-1, pred.size(-1))  # (B*T, vocab_size)
            decoder_loss = decoder_loss.reshape(-1)  
            
            optimizer.zero_grad()
            
            loss = loss_fn(pred ,decoder_loss)
            # print(loss)
            total_loss += loss
            
            
            # print(f' Batch_elemnents  :    {batch}    loss : {loss} ')
            
            # if not torch.isfinite(loss):
            #     print("Loss exploded")
            #     print("max logit:", pred.max().item())
            #     print("min logit:", pred.min().item())
            #     break
            
            total_batch += 1
            
            loss.backward()
            
            optimizer.step()
            
        print(f' Epochs   :    {epoch}         loss :  {total_loss/total_batch} ')
        
        torch.save(model.state_dict() ,'practice_model/100k_model_06.pth')
            
            
            
        
            