import sentencepiece as spm
import torch
from torch import nn
from torch.utils.data import Dataset ,DataLoader
from datasets import load_dataset


dataset = load_dataset("cfilt/iitb-english-hindi")
train_ds = dataset['train'].select(range(100000))
# train_ds = dataset['train']

# print(dataset['train'][:5])

# with open('practice_model/dataset/validation.txt' ,'w' ,encoding='utf-8') as f :
#     for row in dataset["validation"]:
#         f.write(row["translation"]['en'] + '\n')
#         f.write(row["translation"]['hi'] + '\n')
        
# with open('practice_model/dataset/test.txt' ,'w' ,encoding='utf-8') as f :
#     for row in dataset["test"]:
#         f.write(row["translation"]['en'] + '\n')
#         f.write(row["translation"]['hi'] + '\n')



# spm.SentencePieceTrainer.Train(
#     input = "practice_model/dataset/train.txt",
#     model_prefix ='en_hi_tokenizer',
#     vocab_size =32000,
#     model_type = 'unigram'
# )

sp =spm.SentencePieceProcessor()
sp.Load('practice_model/sp_tokenizer/en_hi_tokenizer.model')
vocab_size = sp.GetPieceSize()
# print(vocab_size)

sp.SetEncodeExtraOptions('bos:eos')

def tokenize(row):
    return{
        'input_ids':sp.Encode(row['translation']['en'],out_type=int),
        'target_ids':sp.Encode(row['translation']['hi'],out_type=int)
        
    }
    
    
train_ds_idx = train_ds.map(tokenize , remove_columns=['translation'])

max_length = 64

def padding_sequence(train_ds_idx):
    
    input_ids =train_ds_idx['input_ids']
    target_ids =train_ds_idx['target_ids']
    
    if len(input_ids) > max_length:
            
        input_ids = input_ids[:max_length-1] + [input_ids[-1]] 
        
    if len(target_ids) > max_length:
            
        target_ids = target_ids[:max_length-1] + [target_ids[-1]] 
        
    if len(input_ids) < max_length:
        input_ids = input_ids + [0]*(max_length - len(input_ids))
        
    if len(target_ids) < max_length:
        target_ids = target_ids + [0]*(max_length - len(target_ids))
        
    return {'input_ids':input_ids , 'target_ids' :target_ids}


pad_train_ds_idx =train_ds_idx.map(padding_sequence)


PAD_IDX = sp.pad_id()
# print(PAD_IDX)

class Mydataset(Dataset):
    def __init__(self , train_dataset):
        
        self.dataset = train_dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        
        item =self.dataset[idx]
        
        return {
            "input_ids" : torch.tensor(item['input_ids'],dtype=torch.long),
            "target_ids" : torch.tensor(item['target_ids'],dtype=torch.long)
        }
        
        
train_dataset =Mydataset(pad_train_ds_idx)

dataloader = DataLoader(train_dataset ,batch_size= 32 , shuffle=True)


# print(next(iter(dataloader)).shape)
# x ,y = next(iter(dataloader))
# print(x.shape , y.shape)

# for batch in dataloader:
#     X = batch['input_ids']
#     y= batch['input_ids']
    
#     print(X.shape , y.shape)
#     break