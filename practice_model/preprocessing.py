import torch 
import json
import pandas as pd
import spacy
from torch.utils.data import Dataset,DataLoader

device = 'cuda'  if torch.cuda.is_available else 'cpu'

nlp_en = spacy.blank("en")
nlp_hi = spacy.blank("hi")


path ="practice_model/dataset/Dataset_English_Hindi.csv"
df = pd.read_csv(path)

text_full=(df["English"].astype(str)).tolist()
text =text_full[:1500]

target_full=(df['Hindi'].astype(str)).tolist()
target=target_full[:2000]



path1 ='practice_model/vocab/eng_vocab1.json'

with open(path1, 'r', encoding='utf-8') as f:
    eng_vocab = json.load(f)

path2 ='practice_model/vocab/hin_vocab1.json'

with open(path2, 'r', encoding='utf-8') as f:
    hin_vocab = json.load(f)
    
    
eng_vocab_size =len(eng_vocab)
hin_vocab_size =len(hin_vocab)

print(eng_vocab_size)
print(hin_vocab_size)

def tokenize(text_list,nlp):
    tokenized_sent =[]
    for sent in text_list:
        doc=nlp(sent)
        tokenized_words = []
        
        for token in doc:
            if not token.is_punct and not token.is_quote and not token.is_space:
                tokenized_words.append(token.lower_)
        
        tokenized_sent.append(tokenized_words)
    return tokenized_sent

eng_tokenized_sentence = tokenize(text,nlp_en)
hin_tokenized_sentence = tokenize(target,nlp_hi)



def special_tokens(tokenized_sent_list):
    
    new_sent_list =[]
    
    for sent in tokenized_sent_list:
        
        new_sent_list.append(['<sos>'] + sent + ['<eos>'])

    return new_sent_list
     

hin_special_tokenized_sentence =special_tokens(hin_tokenized_sentence)


def text_to_index(list_sentences,vocab):
    
    tokenize_sentence_to_index =[]
    for sentence in list_sentences:
        sentence_to_index =[]
        tokenize_sentence_to_index.append(sentence_to_index)
        for token in sentence:
            token = vocab[token]
            sentence_to_index.append(token)

    return tokenize_sentence_to_index

eng_index_sentence =text_to_index(eng_tokenized_sentence,eng_vocab)
hin_index_sentence =text_to_index(hin_special_tokenized_sentence,hin_vocab)

# print(hin_index_sentence[70:80])

def size(text):
    size_list = []
    for row in text:
    
        size_list.append(len(row))
    return max(size_list)

eng_max_length = size(eng_tokenized_sentence)
# eng_max_length = 25
print(eng_max_length)
hin_max_length = size(hin_tokenized_sentence)
# hin_max_length = 25
print(hin_max_length)



def padding(max_length,train_sequences):
    
    padded_sequences = []
    max_size = max_length
    
    for sequence in train_sequences:
        sequence = sequence[-max_size:]
        padded_sequences.append(sequence + [0]*((max_size)-len(sequence)) )
    
    return padded_sequences

padded_eng_sent_list = padding(eng_max_length,eng_index_sentence) ## this goes in the embedding layer 
padded_hin_sent_list = padding(hin_max_length,hin_index_sentence) ## this goes in the embedding layer 
PAD_IDX = 0



english_sequences =torch.tensor(padded_eng_sent_list, dtype=torch.long)
hindi_sequences =torch.tensor(padded_hin_sent_list, dtype=torch.long)




class MyDataset(Dataset):
    
    def __init__(self ,x,y):
        super().__init__()
        self.X = x
        self.y =y
        
    def __len__(self):
       
        return self.X.shape[0]
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
my_dataset = MyDataset(english_sequences ,hindi_sequences)

dataloader =DataLoader(dataset= my_dataset,batch_size=32,shuffle=True)

# x ,y = next(iter(dataloader))

# print(y)