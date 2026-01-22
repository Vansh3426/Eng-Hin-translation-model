import torch 
from preprocessing import tokenize , text_to_index ,special_tokens ,nlp_en ,eng_vocab , padding ,hin_vocab ,eng_vocab_size ,hin_vocab_size,eng_max_length,hin_max_length
from decoder import Transformer_runner 


model =Transformer_runner(eng_vocab_size =eng_vocab_size , enc_head_num = 10 ,encoder_embed_dim =128 ,
                           encoder_ff_dim =512 ,eng_max_length = eng_max_length,
                           
                           hin_vocab_size =hin_vocab_size ,dec_head_num = 10 , decoder_embed_dim  = 128,
                           decoder_ff_dim = 512, hin_max_length = hin_max_length)

model.load_state_dict(torch.load('practice_model/model_01.pth'))
model.eval()



def prediction(text ,model):
    
    tokenize_text = tokenize(text,nlp_en)
    
    max_length =50
    indexed_sent =text_to_index(tokenize_text,eng_vocab)
    
    # padded_sent =padding(max_length ,indexed_sent)
    
    index2token = {idx: token for token, idx in hin_vocab.items()}

    encoder_input = torch.tensor(indexed_sent)
    # print(encoder_input.shape)
    decoder_input = torch.tensor([[hin_vocab["<sos>"]]])
    # print(decoder_input.shape)
    output =[]
    
    for i in range(max_length):
        
        pred = model(encoder_input , decoder_input)
        torch.set_printoptions(threshold=torch.inf)
        # print(pred[:, -1, :])
        torch.set_printoptions(profile="default")
        next_token = torch.argmax(pred[:,-1,:] , dim=-1).item()
        # print(next_token)
        next_token_tensor = torch.tensor([[next_token]])
        decoder_input = torch.cat([decoder_input, next_token_tensor], dim=1)
        
        
        if next_token == hin_vocab['<eos>']:
            break
        output.append(index2token[next_token])
        
    return output
    
    
    




text = ["I forgot "]

output = prediction(text ,model)

sentence =" ".join(output)
print(sentence)