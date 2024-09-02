import torch 
import torch.nn as nn

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

class MMoE(torch.nn.Module):
    def __init__(self, cat_dim_list, num_dim, emb_dim, num_experts, bottom_dim, tower_dim, expert_num, dropout):
        super(MMoE, self).__init__()
        self.Embedding = nn.ModuleList(
            [nn.Embedding(cat_dim_list[i], emb_dim) for i in range(len(cat_dim_list))]
        )
        self.num_layer = nn.Linear(num_dim, emb_dim)
        self.embed_output_dim = (1 + len(cat_dim_list)) * emb_dim
        self.expert_num = expert_num 
        
        self.expert = nn.ModuleList([FeedForward(emb_dim, bottom_dim, dropout) for i in range(expert_num)])
        self.gate = nn.ModuleList([nn.Sequential(nn.Linear(self.embed_output_dim, expert_num), nn.Softmax(dim=1)) for i in range(expert_num)])
        self.tower = nn.ModuleList([FeedForward(bottom_dim, tower_dim, 1, dropout) for i in range(expert_num)])
    
    def forward(self, cat_x, num_x):
        cat_emb = self.Embedding(cat_x)
        num_emb = self.num_layer(num_x)
        emb = torch.cat([cat_emb, num_emb], dim=1).view(-1, self.embed_output_dim) # batch_size x output_dim
        gate_value = [self.gate[i](emb) for i in range(self.expert_num)]
        fea = torch.cat([self.expert[i](emb).unsqueeze(1) for i in range(self.expert_num)], dim=1) # batch_size x (expert_num x bottom_dim)
        task_feat = [torch.bmm(gate_value[i], fea).squeeze(1) for i in range(self.expert_num)]
        
        results = [torch.sigmoid(self.tower[i](task_feat[i])) for i in range(self.expert_num)]
        return results