import numpy as np
import torch
# import argparse
# from peft import LoraConfig, get_peft_model
# from peft.tuners.lora.layer import MultiheadAttention as PeftMha
import torch.nn.functional as F

class CrossEntropyWithMargin():
    def __init__(self, margin) -> None:
        self.margin = margin
        self.base_criterion  = torch.nn.CrossEntropyLoss(reduction='none')
    def cal_loss(self, pred, true): ## expected input 1024 x 3 ### sequence matters !
        true_indices = torch.argmax(true, dim=1)
        predicted_values = pred.gather(1, true_indices.unsqueeze(1)).squeeze()
        true_values = true.gather(1, true_indices.unsqueeze(1)).squeeze()   
        margin = torch.abs(true_values - predicted_values)

        loss = self.base_criterion(pred, true)
        mask = margin < self.margin
        loss = loss * ~mask
        average_loss = loss.mean()
        return average_loss


class GatingLayer(torch.nn.Module):
    def __init__(self, num_experts):
        super(GatingLayer, self).__init__()
        self.num_experts = num_experts
        #self.gate = torch.nn.Linear(200, 1) ##TODO experiment differnet gating mechanisms
        self.gate = torch.nn.Linear(10000,1)
    
    def forward(self, expert_outputs):
        ### expert_outputs of shape 1024 x 3 x 200 x 50
        expert_outputs = expert_outputs.flatten(start_dim = 2)
        raw_weights = self.gate(expert_outputs).squeeze(2)
        scaled_weights = F.softmax(raw_weights, dim=-1)
        return scaled_weights

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate, use_conv):

        super(PointWiseFeedForward, self).__init__()
        self.use_conv = use_conv

        if self.use_conv:
            self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
            self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        else:
            self.linear1 = torch.nn.Linear(hidden_units, hidden_units)
            self.linear2 = torch.nn.Linear(hidden_units, hidden_units)
        
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        if self.use_conv:
            outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
            outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
            outputs += inputs
            return outputs
        else:
            outputs = self.dropout2(self.linear2(self.relu(self.dropout1(self.linear1(inputs)))))
            outputs += inputs
            return outputs

class SASRecMoE(torch.nn.Module):
    def __init__(self, user_num, item_num, hard_gates, num_experts, user_groups ,args):
        super(SASRecMoE, self).__init__()

        self.num_experts = num_experts
        self.user_groups = user_groups
        self.user_gate = {}
        assert self.num_experts == len(np.unique(np.array(list(user_groups.values()))))
        for user_type in np.unique(np.array(list(user_groups.values()))):
            tmp = torch.zeros(self.num_experts, device=args.device)
            index = int(user_type.split('_')[1])-1
            tmp[index] = 1
            self.user_gate[user_type] = tmp.float()
        
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
        self.aux_loss_module = CrossEntropyWithMargin(margin=0.2)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.Linear(1,1) ### will be replaced afterwards
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate, args.use_conv)
            self.forward_layers.append(new_fwd_layer)

            new_gating_layer = GatingLayer(num_experts=self.num_experts)
            self.attention_gates.append(new_gating_layer)



    def log2feats(self, log_seqs, user_id):
        aux_loss = 0
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        poss *= (log_seqs != 0)
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)
        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):

            usertype = [self.user_groups[str(i)] for i in user_id]

            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            # mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
            #                                 attn_mask=attention_mask)

            ### Experts Attention
            mha_outputs_from_experts = list()
            for expert in self.attention_layers[i]:
                mha_outputs_expert, _  = expert(Q, seqs, seqs, attn_mask=attention_mask)
                mha_outputs_from_experts.append(mha_outputs_expert)
            mha_outputs = (torch.stack(mha_outputs_from_experts).permute(2,0,1,3)) # batch x 3 x 200 x 50

            if self.hard_gates:
                usergate = torch.stack([self.user_gate[i] for i in usertype]) ### 1024 x 3
                aux_loss = 0
            else:
                usergate = self.attention_gates[i](mha_outputs) ### 1024 x 3
                usergate_true = torch.stack([self.user_gate[i] for i in usertype]) ### 1024 x 3
                aux_loss = aux_loss + self.aux_loss_module.cal_loss(pred=usergate, true=usergate_true)

            usergate = usergate.view(-1,self.num_experts,1,1)
            mha_outputs = (usergate * mha_outputs).sum(dim=1).permute(1,0,2)
                                                                
            ###
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)
        return log_feats, aux_loss

    def forward(self, user_id, log_seqs, pos_seqs, neg_seqs): # for training        
        log_feats, aux_loss = self.log2feats(log_seqs, user_id) # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits, aux_loss

    def predict(self, user_id, log_seqs, item_indices): 
        log_feats, aux_loss = self.log2feats(log_seqs, user_id)

        final_feat = log_feats[:, -1, :]

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits

    def predict_helper(self, user_id, log_seqs, item_indices): 
        log_feats, aux_loss = self.log2feats(log_seqs, user_id)
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        return final_feat, item_embs

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', default='ml-1m',)
#     parser.add_argument('--train_dir', default='default',)
#     parser.add_argument('--batch_size', default=1024, type=int)
#     parser.add_argument('--lr', default=5e-4, type=float)
#     parser.add_argument('--maxlen', default=200, type=int)
#     parser.add_argument('--hidden_units', default=50, type=int)
#     parser.add_argument('--num_blocks', default=2, type=int)
#     parser.add_argument('--num_epochs', default=25, type=int)
#     parser.add_argument('--num_heads', default=1, type=int)
#     parser.add_argument('--dropout_rate', default=0.2, type=float)
#     parser.add_argument('--l2_emb', default=0.0, type=float)
#     parser.add_argument('--device', default='cpu', type=str)
#     args = parser.parse_args()

#     model = SASRecMoE(6040,3416, False, args)
#     for name, param in model.named_parameters():
#         if "gate" in name:
#             param.requires_grad = True
#         else:
#             param.requires_grad = False
    
#     for name, param in model.named_parameters():
#         trainable_params = 0
#         all_param = 0
#         for _, param in model.named_parameters():
#             all_param += param.numel()
#             if param.requires_grad:
#                 trainable_params += param.numel()
#     print(
#         f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
#     )
#     test = torch.randn((1024,3,200,50))
#     testgate = model.attention_gates[0]
#     out = testgate(test)
#     print(out.shape)

    


