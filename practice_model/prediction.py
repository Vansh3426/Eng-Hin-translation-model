import torch 
from preprocessing import tokenize , text_to_index  ,nlp_en ,padding
from decoder import Transformer_runner 
import json




path1 ='practice_model/vocab/eng_vocab3.json'

with open(path1, 'r', encoding='utf-8') as f:
    eng_vocab = json.load(f)

path2 ='practice_model/vocab/hin_vocab3.json'

with open(path2, 'r', encoding='utf-8') as f:
    hin_vocab = json.load(f)
    
    

eng_vocab_size =len(eng_vocab)
hin_vocab_size =len(hin_vocab)

# print(eng_vocab_size)
# print(hin_vocab_size)


model =Transformer_runner(eng_vocab_size =eng_vocab_size , enc_head_num = 8 ,encoder_embed_dim =128 ,
                           encoder_ff_dim =512 ,eng_max_length = 50,
                           
                           hin_vocab_size =hin_vocab_size ,dec_head_num = 8, decoder_embed_dim  = 128,
                           decoder_ff_dim = 512, hin_max_length = 50)

model.load_state_dict(torch.load('practice_model/model_04.pth'))
model.eval()



def prediction(text ,model):
    
    tokenize_text = tokenize(text,nlp_en)
    
    max_length =50
    indexed_sent =text_to_index(tokenize_text,eng_vocab)
    
    # pad_len = max_length - len(indexed_sent)
    # pad_tensor =torch.tensor([eng_vocab["<pad>"]] * pad_len)
    # indexed_sent = torch.tensor(indexed_sent).squeeze()
    
    # indexed_sent = torch.cat((indexed_sent,pad_tensor),dim=0)
    
    index2token = {idx: token for token, idx in hin_vocab.items()}
    # print(indexed_sent)
    encoder_input = torch.tensor(indexed_sent)
    # print(encoder_input.shape)
    decoder_input = torch.tensor([[hin_vocab["<sos>"]]])
    # print(decoder_input.shape)
    output =[]
    
    for i in range(max_length):
        
        pred = model(encoder_input , decoder_input)
        # torch.set_printoptions(threshold=torch.inf)
        # print(pred[:, -1, :])
        # torch.set_printoptions(profile="default")
        next_token = torch.argmax(pred[:,-1,:] , dim=-1).item()
        # print(next_token)
        next_token_tensor = torch.tensor([[next_token]])
        # print(next_token_tensor)
        decoder_input = torch.cat([decoder_input, next_token_tensor], dim=1)
        # print(decoder_input)
        
        
        if next_token == hin_vocab['<eos>']:
            break
        output.append(index2token[next_token])
        
    return output
    
    
    
    
# def prediction(text, model):
#     device = next(model.parameters()).device

#     tokenize_text = tokenize(text, nlp_en)
#     indexed_sent = text_to_index(tokenize_text, eng_vocab)

#     max_length = 50
#     indexed_sent = indexed_sent[:max_length]
#     pad_len = max_length - len(indexed_sent)
#     indexed_sent = indexed_sent + [eng_vocab["<pad>"]] * pad_len

#     encoder_input = torch.tensor(indexed_sent).unsqueeze(0).to(device)

#     decoder_input = torch.tensor([[hin_vocab["<sos>"]]], device=device)

#     index2token = {idx: token for token, idx in hin_vocab.items()}
#     output = []

#     model.eval()
#     with torch.no_grad():
#         for _ in range(max_length):
#             pred = model(encoder_input, decoder_input)
#             logits = pred[:, -1, :]
#             logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)

#             next_token = logits.argmax(dim=-1).item()
#             decoder_input = torch.cat(
#                 [decoder_input, torch.tensor([[next_token]], device=device)],
#                 dim=1
#             )

#             if next_token == hin_vocab["<eos>"]:
#                 break

#             output.append(index2token.get(next_token, ""))

#     return output

    
    




text = ["I hear that she's a famous actress."]

output = prediction(text ,model)
print(output)
sentence =" ".join(output)
print(sentence)