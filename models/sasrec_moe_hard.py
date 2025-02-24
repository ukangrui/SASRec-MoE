import numpy as np
import torch
import torch.nn.functional as F
import sys


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
    
class SASRecMoEHard(torch.nn.Module):
    def __init__(self, user_num, item_num, hidden_units, maxlen, num_blocks, num_heads, dropout_rate, device, grouping_file):
        super(SASRecMoEHard, self).__init__()

        self.hard_user_gate = {}

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
        
        # assert self.num_experts == len(np.unique(np.array(list(grouping_file.values()))))
        self.grouping_file = grouping_file
        self.num_experts= len(np.unique(np.array(list(grouping_file.values()))))
        for user_type in np.unique(np.array(list(grouping_file.values()))):
            tmp = torch.zeros(self.num_experts, device=device)
            index = int(user_type.split('_')[1])-1
            tmp[index] = 1
            self.hard_user_gate[user_type] = tmp.float()


    def log2feats(self, log_seqs, user_ids):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        poss = torch.from_numpy(poss).to(log_seqs.device)
        poss *= (log_seqs != 0)
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)
        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        user_ids = [i.item() for i in user_ids]
        usertypes = [self.grouping_file[str(i)] for i in user_ids]

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

            usergate = torch.stack([self.hard_user_gate[i] for i in usertypes]) ### batch x num_experts


            usergate = usergate.view(-1,self.num_experts,1,1)
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
