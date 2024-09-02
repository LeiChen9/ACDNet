'''
Author: LeiChen9 chenlei9691@gmail.com
Date: 2024-08-31 14:10:40
LastEditors: LeiChen9 chenlei9691@gmail.com
LastEditTime: 2024-09-01 17:49:51
FilePath: /Poutry/Users/lei/Documents/Code/ACDNet/model/ACDNet.py
Description: 

Copyright (c) 2024 by Riceball, All Rights Reserved. 
'''
import torch 
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class Head(nn.Module):
    def __init__(self, input_dim, embed_dim, dropout):
        super().__init__()
        self.q = nn.Linear(input_dim, embed_dim)
        self.k = nn.Linear(input_dim, embed_dim)
        self.v = nn.Linear(input_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        wei = q @ k.transpose(-2, -1) * C**0.5
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v 
        return out
        

class ACDNet(torch.nn.Module):
    def __init__(self, cat_dim_list, num_dim, emb_dim, bottom_dim, tower_dim, expert_num, dropout):
        super(ACDNet, self).__init__()
        self.Embedding = nn.ModuleList(
            [nn.Embedding(cat_dim_list[i], emb_dim) for i in range(len(cat_dim_list))]
        )
        self.num_layer = nn.Linear(num_dim, emb_dim)
        self.embed_output_dim = (1 + len(cat_dim_list)) * emb_dim
        self.expert_num = expert_num 
        
        self.expert = nn.ModuleList([FeedForward(emb_dim, bottom_dim, dropout) for i in range(expert_num)])
        self.gate = nn.ModuleList([nn.Sequential(nn.Linear(self.embed_output_dim, expert_num), nn.Softmax(dim=1)) for i in range(expert_num)])
        self.tower = nn.ModuleList([FeedForward(bottom_dim, tower_dim, 1, dropout) for i in range(expert_num)])
        
        self.query = nn.ModuleList([Head(self.embed_output_dim, emb_dim / expert_num, dropout) for i in range(expert_num)])
        self.key = nn.ModuleList([Head(self.embed_output_dim, emb_dim / expert_num, dropout) for i in range(expert_num)])
        self.value = nn.ModuleList([Head(self.embed_output_dim, emb_dim / expert_num, dropout) for i in range(expert_num)])
    
    def forward(self, cat_x, num_x):
        cat_emb = self.Embedding(cat_x)
        num_emb = self.num_layer(num_x)
        emb = torch.cat([cat_emb, num_emb], dim=1).view(-1, self.embed_output_dim) # batch_size x output_dim
        
        attn_emb = []
        for i in range(len(self.expert_num)):
            q = self.query[i](emb)
            k = self.key[i](emb)
            wei = q @ k.transpose(-2, -1) * self.emb_dim**0.5
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            out_list = []
            for j in range(len(self.expert_num)):
                v = self.value[j](emb)
                out = wei @ v
                out_list.append(out)
            out = torch.cat(out_list, dim=1)
            attn_emb.append(out)
                
        gate_value = [self.gate[i](attn_emb[i]) for i in range(self.expert_num)]
        fea = torch.cat([self.expert[i](attn_emb[i]).unsqueeze(1) for i in range(self.expert_num)], dim=1) # batch_size x (expert_num x bottom_dim)
        task_feat = [torch.bmm(gate_value[i], fea).squeeze(1) for i in range(self.expert_num)]
        
        results = [torch.sigmoid(self.tower[i](task_feat[i])) for i in range(self.expert_num)]
        return results