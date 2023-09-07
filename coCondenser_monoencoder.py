import pandas as pd 
import numpy as np 
from transformers import * 
from tqdm.auto import tqdm 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from pytorch_metric_learning import miners, losses, distances 
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
class custom_collate(object): 
    def __init__(self, plm="Luyu/co-condenser-marco"): 
        self.tokenizer = AutoTokenizer.from_pretrained(plm)
        self.chunk_size = 512 
    def __call__(self, batch):
        label_dict = {"A":0, "B":1, "C":2, "D":3, "E":4}
        prompt_input_ids, prompt_attn_masks, prompt_labels = [], [], [] 
        A_input_ids, A_attn_masks, A_labels = [], [], [] 
        B_input_ids, B_attn_masks, B_labels = [], [], [] 
        C_input_ids, C_attn_masks, C_labels = [], [], [] 
        D_input_ids, D_attn_masks, D_labels = [], [], [] 
        E_input_ids, E_attn_masks, E_labels = [], [], [] 
        answers = [] 
        ids = 0 
        for idx, row in enumerate(batch): 
            _, prompt, A, B, C, D, E, answer = row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7] 
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
            
            prompt_labels.append(ids) 
            
            answers.append(label_dict[answer]) 
            
            if answer == "A":
                A_labels.append(ids) 
                B_labels.append(ids+1) 
                C_labels.append(ids+2) 
                D_labels.append(ids+3) 
                E_labels.append(ids+4)             
            elif answer == "B":
                A_labels.append(ids+1) 
                B_labels.append(ids) 
                C_labels.append(ids+2) 
                D_labels.append(ids+3) 
                E_labels.append(ids+4)
            elif answer == "C":
                A_labels.append(ids+1) 
                B_labels.append(ids+2) 
                C_labels.append(ids) 
                D_labels.append(ids+3) 
                E_labels.append(ids+4)
            elif answer == "D":
                A_labels.append(ids+1) 
                B_labels.append(ids+2) 
                C_labels.append(ids+3) 
                D_labels.append(ids) 
                E_labels.append(ids+4)             
            elif answer == "E":
                A_labels.append(ids+1) 
                B_labels.append(ids+2) 
                C_labels.append(ids+3) 
                D_labels.append(ids+4) 
                E_labels.append(ids)             
            ids += 5
        prompt_input_ids = torch.stack(prompt_input_ids, dim=0).squeeze(dim=1) 
        prompt_attn_masks = torch.stack(prompt_attn_masks, dim=0).squeeze(dim=1) 
        prompt_labels = torch.tensor(prompt_labels, dtype=int) 
        
        A_input_ids = torch.stack(A_input_ids, dim=0).squeeze(dim=1) 
        A_attn_masks = torch.stack(A_attn_masks, dim=0).squeeze(dim=1) 
        A_labels = torch.tensor(A_labels, dtype=int) 
        
        B_input_ids = torch.stack(B_input_ids, dim=0).squeeze(dim=1) 
        B_attn_masks = torch.stack(B_attn_masks, dim=0).squeeze(dim=1) 
        B_labels = torch.tensor(B_labels, dtype=int) 
        
        C_input_ids = torch.stack(C_input_ids, dim=0).squeeze(dim=1) 
        C_attn_masks = torch.stack(C_attn_masks, dim=0).squeeze(dim=1) 
        C_labels = torch.tensor(C_labels, dtype=int) 
        
        D_input_ids = torch.stack(D_input_ids, dim=0).squeeze(dim=1) 
        D_attn_masks = torch.stack(D_attn_masks, dim=0).squeeze(dim=1) 
        D_labels = torch.tensor(D_labels, dtype=int) 
        
        E_input_ids = torch.stack(E_input_ids, dim=0).squeeze(dim=1) 
        E_attn_masks = torch.stack(E_attn_masks, dim=0).squeeze(dim=1) 
        E_labels = torch.tensor(E_labels, dtype=int) 
        
        answers = torch.tensor(answers, dtype=int)
        
        return prompt_input_ids, prompt_attn_masks, prompt_labels, A_input_ids, A_attn_masks, A_labels, B_input_ids, B_attn_masks, B_labels, C_input_ids, C_attn_masks, C_labels, D_input_ids, D_attn_masks, D_labels, E_input_ids, E_attn_masks, E_labels, answers 
    

class Embedder(nn.Module): 
    def __init__(self, plm="Luyu/co-condenser-marco"):
        super(Embedder, self).__init__() 
        self.model = AutoModel.from_pretrained(plm) 
    def forward(self, input_ids, attn_masks):
        x = self.model(input_ids, attn_masks)[0][:, 0, :] # CLS-Pooling 
        return x 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 

X = train[["prompt", "A", "B", "C", "D", "E"]]
y = train["answer"] 

foldwise_map_scores = [] 

for idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    if idx > 0: 
        break 
    print(f"===== validating on fold {idx+1} =====")
    train_df, valid_df = train.iloc[train_idx], train.iloc[val_idx] 
    train_set = PairData(train_df) 
    valid_set = PairData(valid_df)
    collate = custom_collate() 
    train_dataloader = DataLoader(train_set, batch_size=32,collate_fn=collate, shuffle=False) 
    valid_dataloader = DataLoader(valid_set, batch_size=1,collate_fn=collate, shuffle=False) 
    
    model = Embedder()
    model.to(device) 
    
    model_tokenizer = AutoTokenizer.from_pretrained("Luyu/co-condenser-marco") 
    
    epochs = 20 
    params = list(model.parameters()) 
    optimizer = torch.optim.AdamW(params, lr=2e-5, eps=1e-8) 
    t_total = len(train_dataloader) * epochs 
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05 * t_total), num_training_steps=t_total) 
    miner = miners.MultiSimilarityMiner() # distance function: cosine similarity 
    loss_func = losses.MultiSimilarityLoss() # distance function: cosine similarity 
    train_losses, validation_losses, validation_MAPs = [], [], []  
    model.zero_grad() 
    torch.cuda.empty_cache() 
    
    best_map_3 = 0
    
    for epoch_i in tqdm(range(0, epochs), desc="Epochs", position=0, leave=True, total=epochs):
        # training 
        train_loss = 0 
        model.train() 
        with tqdm(train_dataloader, unit="batch") as tepoch:
            for step, batch in enumerate(tepoch): 
                batch = tuple(t.to(device) for t in batch) 
                prompt_input_ids, prompt_attn_masks, prompt_labels, A_input_ids, A_attn_masks, A_labels, B_input_ids, B_attn_masks, B_labels, C_input_ids, C_attn_masks, C_labels, D_input_ids, D_attn_masks, D_labels, E_input_ids, E_attn_masks, E_labels, answer = batch
                prompt_embeddings = model(prompt_input_ids, prompt_attn_masks) 
                A_embeddings = model(A_input_ids, A_attn_masks) 
                B_embeddings = model(B_input_ids, B_attn_masks) 
                C_embeddings = model(C_input_ids, C_attn_masks) 
                D_embeddings = model(D_input_ids, D_attn_masks) 
                E_embeddings = model(E_input_ids, E_attn_masks) 
                full_embeddings = torch.cat((prompt_embeddings, A_embeddings, B_embeddings, C_embeddings, D_embeddings, E_embeddings), dim=0) 
                full_labels = torch.cat((prompt_labels, A_labels, B_labels, C_labels, D_labels, E_labels)) 
                hard_pairs = miner(full_embeddings, full_labels) 
                loss = loss_func(full_embeddings, full_labels, hard_pairs) 
                train_loss += loss.item() 
                loss.backward() 
                torch.nn.utils.clip_grad_norm_(params, 1.0) 
                optimizer.step() 
                scheduler.step() 
                model.zero_grad() 
                tepoch.set_postfix(loss=train_loss / (step+1)) 
                time.sleep(0.1)  
        avg_train_loss = train_loss / len(train_dataloader) 
        # validation 
        valid_loss = 0 
        model.eval() 
        for step, batch in tqdm(enumerate(valid_dataloader), position=0, leave=True, total=len(valid_dataloader), desc="validating"):
            batch = tuple(t.to(device) for t in batch)
            prompt_input_ids, prompt_attn_masks, prompt_labels, A_input_ids, A_attn_masks, A_labels, B_input_ids, B_attn_masks, B_labels, C_input_ids, C_attn_masks, C_labels, D_input_ids, D_attn_masks, D_labels, E_input_ids, E_attn_masks, E_labels, answer = batch
            with torch.no_grad():
                prompt_embeddings = model(prompt_input_ids, prompt_attn_masks) 
                A_embeddings = model(A_input_ids, A_attn_masks) 
                B_embeddings = model(B_input_ids, B_attn_masks) 
                C_embeddings = model(C_input_ids, C_attn_masks) 
                D_embeddings = model(D_input_ids, D_attn_masks) 
                E_embeddings = model(E_input_ids, E_attn_masks) 
                full_embeddings = torch.cat((prompt_embeddings, A_embeddings, B_embeddings, C_embeddings, D_embeddings, E_embeddings), dim=0) 
                full_labels = torch.cat((prompt_labels, A_labels, B_labels, C_labels, D_labels, E_labels)) 
                loss = loss_func(full_embeddings, full_labels) 
                valid_loss += loss.item() 
        avg_valid_loss = valid_loss / len(valid_dataloader) 
        # calculate MAP@3 
        with torch.no_grad():
            ap_3 = 0
            model.eval() 
            for step, batch in tqdm(enumerate(valid_dataloader), position=0, leave=True, total=len(valid_dataloader), desc="calculating validation MAP@3"):
                prompt_input_ids, prompt_attn_masks, prompt_labels, A_input_ids, A_attn_masks, A_labels, B_input_ids, B_attn_masks, B_labels, C_input_ids, C_attn_masks, C_labels, D_input_ids, D_attn_masks, D_labels, E_input_ids, E_attn_masks, E_labels, answer = batch
                prompt_embeddings = model(prompt_input_ids.to(device), prompt_attn_masks.to(device)) 
                A_embeddings = model(A_input_ids.to(device), A_attn_masks.to(device)) 
                B_embeddings = model(B_input_ids.to(device), B_attn_masks.to(device)) 
                C_embeddings = model(C_input_ids.to(device), C_attn_masks.to(device)) 
                D_embeddings = model(D_input_ids.to(device), D_attn_masks.to(device)) 
                E_embeddings = model(E_input_ids.to(device), E_attn_masks.to(device)) 
                
                sim_A = F.cosine_similarity(prompt_embeddings, A_embeddings).item() 
                sim_B = F.cosine_similarity(prompt_embeddings, B_embeddings).item()  
                sim_C = F.cosine_similarity(prompt_embeddings, C_embeddings).item()  
                sim_D = F.cosine_similarity(prompt_embeddings, D_embeddings).item()  
                sim_E = F.cosine_similarity(prompt_embeddings, E_embeddings).item()  
                
                cos_scores = torch.Tensor([sim_A, sim_B, sim_C, sim_D, sim_E]) 
                _, sorted_indices = torch.sort(cos_scores, descending=True) 
                
                sorted_indices = sorted_indices[:3] # get top 3 
                for j in range(len(sorted_indices)): 
                    if sorted_indices[j] == answer.item(): 
                        ap_3 += 1 
            map_3 = ap_3 / len(valid_dataloader) 
            
        print(f"Epochs: {epoch_i+1} | training loss: {avg_train_loss} | validation loss: {avg_valid_loss} | validation MAP@3: {map_3}") 
        train_losses.append(avg_train_loss) 
        validation_losses.append(avg_valid_loss) 
        if best_map_3 < map_3: 
            best_map_3 = map_3 
            torch.save(model.state_dict(), f"coCondenser_encoder_fold{idx+1}.pt") 
    
    foldwise_map_scores.append(best_map_3)     
