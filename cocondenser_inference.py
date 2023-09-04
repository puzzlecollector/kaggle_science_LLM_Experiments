import pandas as pd 
import numpy as np 
from transformers import * 
from tqdm.auto import tqdm 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler, IterableDataset 
import time 
import os 
import random 
import math 
from sklearn.model_selection import KFold, StratifiedKFold

train = pd.read_csv("train.csv") 
test = pd.read_csv("test.csv") 
submission = pd.read_csv("sample_submission.csv") 

class PairData(Dataset): 
    def __init__(self, df):
        super(PairData, self).__init__() 
        self.data = df 
    def __getitem__(self, index): 
        return self.data.iloc[index] 
    def __len__(self): 
        return self.data.shape[0] 
    
# create dataloader 
class custom_collate_test(object): 
    def __init__(self, plm="Luyu/co-condenser-marco"): 
        self.tokenizer = AutoTokenizer.from_pretrained(plm)
        self.chunk_size = 512 
    def __call__(self, batch):
        prompt_input_ids, prompt_attn_masks = [], [] 
        A_input_ids, A_attn_masks = [], []
        B_input_ids, B_attn_masks = [], []
        C_input_ids, C_attn_masks = [], []
        D_input_ids, D_attn_masks = [], []
        E_input_ids, E_attn_masks = [], []
        for idx, row in enumerate(batch): 
            _, prompt, A, B, C, D, E = row[0], row[1], row[2], row[3], row[4], row[5], row[6] 
            encoded_prompt = self.tokenizer(prompt, max_length=self.chunk_size, padding="max_length", truncation=True, return_tensors="pt")
            encoded_A = self.tokenizer(A, max_length=self.chunk_size, padding="max_length", truncation=True, return_tensors="pt") 
            encoded_B = self.tokenizer(B, max_length=self.chunk_size, padding="max_length", truncation=True, return_tensors="pt") 
            encoded_C = self.tokenizer(C, max_length=self.chunk_size, padding="max_length", truncation=True, return_tensors="pt") 
            encoded_D = self.tokenizer(D, max_length=self.chunk_size, padding="max_length", truncation=True, return_tensors="pt") 
            encoded_E = self.tokenizer(E, max_length=self.chunk_size, padding="max_length", truncation=True, return_tensors="pt") 
            prompt_input_ids.append(encoded_prompt["input_ids"]) 
            prompt_attn_masks.append(encoded_prompt["attention_mask"]) 
                
            A_input_ids.append(encoded_A["input_ids"]) 
            A_attn_masks.append(encoded_A["attention_mask"]) 
                
            B_input_ids.append(encoded_B["input_ids"]) 
            B_attn_masks.append(encoded_B["attention_mask"]) 
                
            C_input_ids.append(encoded_C["input_ids"]) 
            C_attn_masks.append(encoded_C["attention_mask"]) 
                
            D_input_ids.append(encoded_D["input_ids"]) 
            D_attn_masks.append(encoded_D["attention_mask"]) 
                
            E_input_ids.append(encoded_E["input_ids"]) 
            E_attn_masks.append(encoded_E["attention_mask"]) 
            
        prompt_input_ids = torch.stack(prompt_input_ids, dim=0).squeeze(dim=1) 
        prompt_attn_masks = torch.stack(prompt_attn_masks, dim=0).squeeze(dim=1) 
        
        A_input_ids = torch.stack(A_input_ids, dim=0).squeeze(dim=1) 
        A_attn_masks = torch.stack(A_attn_masks, dim=0).squeeze(dim=1) 
        
        B_input_ids = torch.stack(B_input_ids, dim=0).squeeze(dim=1) 
        B_attn_masks = torch.stack(B_attn_masks, dim=0).squeeze(dim=1) 
        
        C_input_ids = torch.stack(C_input_ids, dim=0).squeeze(dim=1) 
        C_attn_masks = torch.stack(C_attn_masks, dim=0).squeeze(dim=1) 
        
        D_input_ids = torch.stack(D_input_ids, dim=0).squeeze(dim=1) 
        D_attn_masks = torch.stack(D_attn_masks, dim=0).squeeze(dim=1) 
        
        E_input_ids = torch.stack(E_input_ids, dim=0).squeeze(dim=1) 
        E_attn_masks = torch.stack(E_attn_masks, dim=0).squeeze(dim=1) 
                
        return prompt_input_ids, prompt_attn_masks, A_input_ids, A_attn_masks, B_input_ids, B_attn_masks, C_input_ids, C_attn_masks, D_input_ids, D_attn_masks, E_input_ids, E_attn_masks 

    
test_set = PairData(test) 
collate = custom_collate_test() 
test_dataloader = DataLoader(test_set, batch_size=1, collate_fn=collate, shuffle=False) 

class Embedder(nn.Module): 
    def __init__(self, plm="Luyu/co-condenser-marco"):
        super(Embedder, self).__init__() 
        self.model = AutoModel.from_pretrained(plm) 
    def forward(self, input_ids, attn_masks):
        x = self.model(input_ids, attn_masks)[0][:, 0, :] # CLS-Pooling 
        return x 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")     

prompt_encoders, answer_encoders = [], [] 

for i in range(5): 
    prompt_chkpt = torch.load(f"cocondenser_prompt_encoder_fold_{i+1}.pt") 
    answer_chkpt = torch.load(f"cocondenser_answer_encoder_fold_{i+1}.pt")
        
    prompt_encoder = Embedder() 
    prompt_encoder.load_state_dict(prompt_chkpt) 
    prompt_encoder.to(device) 
        
    answer_encoder = Embedder() 
    answer_encoder.load_state_dict(answer_chkpt) 
    answer_encoder.to(device) 
    
    prompt_encoders.append(prompt_encoder) 
    answer_encoders.append(answer_encoder) 

    
predictions = [] 

for step, batch in tqdm(enumerate(test_dataloader), position=0, leave=True, total=len(test_dataloader)):
    prompt_input_ids, prompt_attn_masks, A_input_ids, A_attn_masks, B_input_ids, B_attn_masks, C_input_ids, C_attn_masks, D_input_ids, D_attn_masks, E_input_ids, E_attn_masks = batch
    avg_scores = torch.zeros(5) 
    # calculate soft-voting ensemble 
    for i in range(5): 
        prompt_encoder = prompt_encoders[i] 
        answer_encoder = answer_encoders[i] 
        
        prompt_encoder.eval() 
        answer_encoder.eval() 
        
        with torch.no_grad():
            prompt_embeddings = prompt_encoder(prompt_input_ids.to(device), prompt_attn_masks.to(device)) 
            A_embeddings = answer_encoder(A_input_ids.to(device), A_attn_masks.to(device)) 
            B_embeddings = answer_encoder(B_input_ids.to(device), B_attn_masks.to(device)) 
            C_embeddings = answer_encoder(C_input_ids.to(device), C_attn_masks.to(device)) 
            D_embeddings = answer_encoder(D_input_ids.to(device), D_attn_masks.to(device)) 
            E_embeddings = answer_encoder(E_input_ids.to(device), E_attn_masks.to(device)) 
                
            sim_A = F.cosine_similarity(prompt_embeddings, A_embeddings).item() 
            sim_B = F.cosine_similarity(prompt_embeddings, B_embeddings).item()  
            sim_C = F.cosine_similarity(prompt_embeddings, C_embeddings).item()  
            sim_D = F.cosine_similarity(prompt_embeddings, D_embeddings).item()  
            sim_E = F.cosine_similarity(prompt_embeddings, E_embeddings).item()  
            
            cos_scores = torch.Tensor([sim_A, sim_B, sim_C, sim_D, sim_E]) 
            
            avg_scores += cos_scores 
    
    avg_scores /= 5.0 # obtain average scores  
    _, sorted_indices = torch.sort(avg_scores, descending=True) 
    sorted_indices = sorted_indices[:3] 
    d = {0:"A", 1:"B", 2:"C", 3:"D", 4:"E"}         
    prediction = d[sorted_indices[0].item()] + " " + d[sorted_indices[1].item()] + " " + d[sorted_indices[2].item()] 
    predictions.append(prediction) 

submission["prediction"] = predictions 
submission.to_csv("cocondenser_10_epochs.csv", index=False)
