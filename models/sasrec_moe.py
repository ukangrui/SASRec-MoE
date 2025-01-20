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
        self.proj1 = torch.nn.Linear(10000,25)
        self.proj2 = torch.nn.Linear(225, 32)
        self.proj3 = torch.nn.Linear(35, self.out_dim)
    
    def forward(self, base_feat, addi_feat1, addi_feat2):
        return self.proj3(torch.cat((F.relu(self.proj2(torch.cat((F.relu(self.proj1(base_feat)), addi_feat1), dim=1))), addi_feat2), dim=1))



class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.linear1 = torch.nn.Linear(hidden_units, hidden_units)
        self.linear2 = torch.nn.Linear(hidden_units, hidden_units)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.linear2(self.relu(self.dropout1(self.linear1(inputs)))))
        outputs += inputs
        return outputs

class SASRecMoE(torch.nn.Module):
    def __init__(self, user_num, item_num, hard_gates, num_experts, user_groups, lora_weights, args):
        super(SASRecMoE, self).__init__()
        self.lora_weights = lora_weights
        self.num_experts = num_experts
        self.user_groups = user_groups ### user_split dict
        self.hard_user_gate = {}
        assert self.num_experts == len(np.unique(np.array(list(user_groups.values()))))
        for user_type in np.unique(np.array(list(user_groups.values()))):
            tmp = torch.zeros(self.num_experts, device=args.device)
            index = int(user_type.split('_')[1])-1
            tmp[index] = 1
            self.hard_user_gate[user_type] = tmp.float()
        
        self.hard_gates = hard_gates
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() 
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.attention_gates = torch.nn.ModuleList()
        self.trainable_router = Router(self.num_experts)
        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)


    def log2feats(self, log_seqs, user_id):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        poss *= (log_seqs != 0)
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)
        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        
        ### Router
        ###
        gate_logits = self.get_user_repr(log_seqs) ### batch x 
        ###



        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        log_feats = self.last_layernorm(seqs)
        return log_feats


    def forward(self, user_id, log_seqs, pos_seqs, neg_seqs): 
        log_feats, gate_logits = self.log2feats(log_seqs, user_id) 
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        return pos_logits, neg_logits, gate_logits

    def predict(self, user_id, log_seqs, item_indices): 
        log_feats, gate_logits = self.log2feats(log_seqs, user_id)
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits

    def predict_batch(self, user_ids, log_seqs, item_indices):
        log_feats, gate_logits = self.log2feats(log_seqs, user_ids)
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        logits =  torch.einsum('bi,bji->bj', final_feat, item_embs)
        return logits
    
    def get_user_repr(self,  log_seqs):
        mu = 4.573289742684675
        std = 1.0152993033240776
        z = torch.tensor((np.log(np.count_nonzero(log_seqs, axis = 1)) - mu)/std)
        zsq = z ** 2
        ztr = z ** 3
        len_feat = torch.vstack([z, zsq ,ztr]).permute(1,0).float() ## batch , 3
        ###########

        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5

        ########
        pool_feat= torch.mean(seqs, dim=2) ## batch, 200
        ########

        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        poss *= (log_seqs != 0)
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)
        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        #########
        attention_first_output, _ = self.attention_layers[0](
            self.attention_layernorms[0](torch.transpose(seqs, 0, 1)),
            torch.transpose(seqs, 0, 1),
            torch.transpose(seqs, 0, 1),
            attn_mask=attention_mask
        )
        base_feat = torch.flatten((self.attention_layernorms[0](torch.transpose(seqs, 0, 1)) + attention_first_output).permute(1,0,2), start_dim=1) # batch, 10000
        #########
        gate_logits = F.softmax(self.trainable_router(base_feat, pool_feat, len_feat.to(self.dev)),dim=1) ### Q Matrix ### batch x 10
        return gate_logits

    
    def load_mha_weights(self, gate_logits, lora_weights):
        pass





# class GatingLayer(torch.nn.Module):
#     def __init__(self, num_experts, user_embs, gamma, hidden_units):
#         super(GatingLayer, self).__init__()
#         self.num_experts = num_experts
#         self.gamma = gamma
#         self.user_embs  = user_embs.to(torch.device('cuda'))
#         self.gate = torch.nn.Sequential(
#             torch.nn.Linear(hidden_units, 256),
#             torch.nn.ReLU(),
#             torch.nn.Linear(256, num_experts)
#         )
    
#     def forward(self, usertypes):
#         indexes = torch.tensor([int(i.split('_')[1]) - 1 for i in usertypes],device = torch.device('cuda'))
#         x = self.gate(self.user_embs(indexes))
#         # mask_before_softmax = torch.full_like(x, 0, device = torch.device('cuda'))
#         # mask_before_softmax[torch.arange(x.size(0)), indexes] = torch.inf
#         # x -= mask_before_softmax
#         # logits = F.softmax(x, dim=1) * (1-self.gamma)
#         # mask_after_softmax = torch.full_like(x, 0, device = torch.device('cuda'))
#         # mask_after_softmax[torch.arange(x.size(0)), indexes] = self.gamma
#         # logits += mask_after_softmax
#         # return logits
#         return x

#         # ### expert_outputs of shape 1024 x 3 x 200 x 64
#         # expert_outputs = expert_outputs.flatten(start_dim = 2)
#         # raw_weights = self.gate(expert_outputs).squeeze(2)
#         # scaled_weights = F.softmax(raw_weights, dim=-1)
#         # return scaled_weights


# class TaskType(torch.nn.Module):
#     def __init__(self, num_types):
#         super(TaskType, self).__init__()
#         self.proj = torch.nn.Linear(50, 3)
#     def forward(self, x):
#         return softmax_with_temperature(self.proj(x))