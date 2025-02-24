import numpy as np
import torch
import torch.nn.functional as F
import sys

def entropy_loss(gate_weights):
    epsilon = 1e-8
    gate_weights = torch.clamp(gate_weights, min=epsilon)
    entropy = -torch.sum(gate_weights * torch.log(gate_weights), dim=1, keepdim=True)
    return entropy

def balance_loss(gate_weights):
    epsilon = 1e-3
    gate_weights = torch.clamp(gate_weights, min=epsilon)
    return -torch.log(torch.prod(gate_weights, dim=1, keepdim=True)).mean()

def softmax_with_temperature(logits, temperature=1.0):
    scaled_logits = logits / temperature
    probabilities = F.softmax(scaled_logits, dim=-1)
    return probabilities


class Router(torch.nn.Module):
    def __init__(self, num_experts):
        super(Router, self).__init__()
        self.out_dim = num_experts
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(50, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, num_experts)
        )
    
    def forward(self, seqs):
        return F.softmax(self.proj(seqs), dim=-1)


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.linear1 = torch.nn.Linear(hidden_units, hidden_units)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_units, hidden_units)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.linear2(self.relu(self.dropout1(self.linear1(inputs)))))
        outputs += inputs
        return outputs
    

class SASRecMoESoft(torch.nn.Module):
    def __init__(self, user_num, item_num, hidden_units, maxlen, num_blocks, num_heads, dropout_rate, device, num_experts  = 10):
        super(SASRecMoESoft, self).__init__()

        self.router = Router(num_experts=10)
        self.num_experts = num_experts

        self.user_num = user_num
        self.item_num = item_num
        self.dev = torch.device(device)
        self.item_emb = torch.nn.Embedding(self.item_num+1, hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(maxlen+1, hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self.attention_layernorms = torch.nn.ModuleList() 
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(hidden_units, eps=1e-8)
        for _ in range(num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            new_attn_layer =  torch.nn.MultiheadAttention(hidden_units,
                                                            num_heads,
                                                            dropout_rate)
            self.attention_layers.append(new_attn_layer)
            new_fwd_layernorm = torch.nn.LayerNorm(hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)
            new_fwd_layer = PointWiseFeedForward(hidden_units, dropout_rate)
            self.forward_layers.append(new_fwd_layer)
        

    def get_user_gate(self, seqs):### batch x seq_len x dim
        seqs = torch.mean(seqs, dim=1)
        return self.router(seqs)


    def log2feats(self, log_seqs, user_ids):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        ##########
        usergate = self.get_user_gate(seqs)
        usergate = usergate.view(-1,self.num_experts,1,1)
        ##########
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        poss = torch.from_numpy(poss).to(log_seqs.device)
        poss *= (log_seqs != 0)
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)
        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))


        for i in range(len(self.attention_layers)):

            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            # mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
            #                                 attn_mask=attention_mask)

            ### Experts Attention
            mha_outputs_from_experts = list()
            for expert in self.attention_layers[i]:
                mha_outputs_expert, _  = expert(Q, seqs, seqs, attn_mask=attention_mask)
                mha_outputs_from_experts.append(mha_outputs_expert)
            mha_outputs = (torch.stack(mha_outputs_from_experts).permute(2,0,1,3)) # batch x num_experts x 200 x 50


            mha_outputs = (usergate * mha_outputs).sum(dim=1).permute(1,0,2)
                                                                
            ###
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)
            seqs = self.forward_layernorms[i](seqs)

            seqs = self.forward_layers[i](seqs)

        log_feats = self.last_layernorm(seqs) 
        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): 
        log_feats = self.log2feats(log_seqs, user_ids)
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        return pos_logits, neg_logits 

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs, user_ids) # user_ids hasn't been used yet
        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits # preds # (U, I)
    
    def predict_batch(self, user_ids, log_seqs, item_indices):
        log_feats = self.log2feats(log_seqs, user_ids)
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        logits =  torch.einsum('bi,bji->bj', final_feat, item_embs)
        return logits
