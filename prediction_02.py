import torch
import sentencepiece as spm
from preprocessing_02 import max_length ,vocab_size
from decoder import Transformer_runner

device = 'cuda' if torch.cuda.is_available() else 'cpu'

sp = spm.SentencePieceProcessor()
sp.load("practice_model/sp_tokenizer/en_hi_tokenizer.model")


model = Transformer_runner(vocab_size , enc_head_num = 4 ,encoder_embed_dim =128 ,
                           encoder_ff_dim =256 ,max_length = max_length ,dec_head_num =4 , decoder_embed_dim  = 128,
                           decoder_ff_dim = 256).to(device)

model.load_state_dict(torch.load('practice_model/100k_model_06.pth'))

model.eval()

def encode(sentence , max_length):

    ids = sp.Encode(sentence , out_type= int)
    
    ids = [sp.bos_id()] + ids+ [sp.eos_id()]
    
    if len(ids) < max_length :
        
        ids = ids + [0]*(max_length - len(ids))
        
    else :
        ids = ids[max_length-1] + [sp.eos_id()]
    
    
    ids = torch.tensor(ids).unsqueeze(0)
    
    return ids
    


def prediction(ids , max_length):
    
    
    encoder_input = ids
    decoder_input = torch.tensor([[sp.bos_id()]] ,device=device)
    # print(decoder_input.shape)
    
    for _ in range(max_length-1):
        
        logits = model(encoder_input, decoder_input)
        # print(f" logits shape {logits.shape}")
        
        next_token_logits =logits[:,-1,:]
        # print(f" next token logits shape {next_token_logits.shape}")
        next_token = torch.argmax(next_token_logits , dim=-1).unsqueeze(1)
        # print(f" next token  shape {next_token.shape}")
        
        decoder_input = torch.cat([decoder_input , next_token] ,dim=1)
        # print(f" decoder input combined  shape {next_token.shape}")
        
        if next_token.item() == sp.eos_id():
            break
        
    return decoder_input 


def decode(ids):
    
    ids =ids.squeeze().tolist()
    output_ids = []
    
    for i in ids :
        
        
        if i not in {sp.pad_id() ,sp.bos_id(),sp.eos_id()}:
            output_ids.append(i)
        
    output_text = sp.Decode(output_ids)
    
    return output_text
    
    

sentence = "The project is too large for the disc even with the overburn option."

ids = encode( sentence , max_length).to(device)

pred_ids = prediction(ids , max_length).to(device)

output_text = decode(pred_ids)

print(output_text)
