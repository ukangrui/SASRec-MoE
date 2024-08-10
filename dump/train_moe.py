import os
import time
import torch
import argparse
import json
from models.model_lora import SASRec
from utils.base_utils import *

import os
import pytorch_lightning as pl
from peft import LoraConfig, get_peft_model, PeftModel
import copy
from models.model_moe import SASRecMoE

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ml-1m',)
parser.add_argument('--train_dir', default='default',)
parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=25, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cpu', type=str)

args = parser.parse_args()

if __name__ == '__main__':
    pl.seed_everything(3407)
    with open('checkpoints/ml1m_popularity.json', 'r') as json_file:
        user_types = json.load(json_file)
    alltypes = sorted(np.unique(np.array(list(user_types.values()))).tolist(), key=lambda x: int(x.split('_')[1])) ### usersplit group
    u2i_index, i2u_index = build_index(args.dataset)
    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    num_batch = (len(user_train) - 1) // args.batch_size + 1
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    _model = SASRec(usernum, itemnum, args).to(args.device)
    _model.load_state_dict(torch.load('checkpoints/base.pth', map_location=torch.device(args.device)), strict=True)

    _ckpts = []

    for type in alltypes:
        type_dir = 'lora_checkpoint_popu/' + type + '/'
        max_ndcg = max([float(i.split('=')[1]) for i in os.listdir(type_dir)])
        _ckpts.append('lora_checkpoint_popu/' + type + '/' + 'ndcg=' +str(max_ndcg))
    # print(np.mean(np.array(_ndcgs)))
    # sys.exit()
    attention_0_experts = [] 
    attention_1_experts = []
    for ckpt in _ckpts:
        lora_test_model = PeftModel.from_pretrained(copy.deepcopy(_model), ckpt)
        merged = lora_test_model.merge_and_unload() ##TODO
        attention_0_experts.append(merged.attention_layers[0])
        attention_1_experts.append(merged.attention_layers[1])
        del lora_test_model
        del merged
    moe_model = SASRecMoE(usernum, itemnum, hard_gates=True,num_experts=100, args=args).to(args.device)
    moe_model.load_state_dict(torch.load('checkpoints/base.pth', map_location=torch.device(args.device)), strict=False)
    moe_model.attention_layers = torch.nn.ModuleList(
        [
            torch.nn.ModuleList(attention_0_experts),
            torch.nn.ModuleList(attention_1_experts),
        ]
    )
    with torch.no_grad():
        moe_model.eval()
        #t_test = evaluate_split(moe_model, dataset, args, ignore = None,  sampling=False)
        ndcg, ht = evaluate_split(moe_model, dataset, args, user_types_dic=user_types, split='split_99')
        print(ndcg)
        # json_data = json.dumps(ndcg)
        # with open('lora_popularity.json', 'w') as json_file:
        #     json_file.write(json_data)

    def mark_only_gating_trainable(model):
        for name, param in model.named_parameters():
            if "gate" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        for name, param in model.named_parameters():
            trainable_params = 0
            all_param = 0
            for _, param in model.named_parameters():
                all_param += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )
    
    mark_only_gating_trainable(model=moe_model)
        
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(moe_model.parameters(), lr=args.lr, weight_decay=0.01)
    for epoch in range(args.num_epochs):
        moe_model.train()
        for step in range(num_batch): 
            u, seq, pos, neg = sampler.next_batch() 
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits, aux_loss = moe_model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device, requires_grad=True), torch.zeros(neg_logits.shape, device=args.device, requires_grad=True)
            optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices]) + bce_criterion(neg_logits[indices], neg_labels[indices])
            print(loss, aux_loss)
            loss += aux_loss
            loss.backward()
            optimizer.step()
        print('epoch:', epoch, 'loss: ', loss.detach().item())
        
        moe_model.eval()
        with torch.no_grad():
            t_test = evaluate_split(moe_model, dataset, args, ignore = None,  sampling=False)
            print(t_test)
            torch.save(moe_model.state_dict(),f'checkpoint_{epoch}.pt')
            




        

    
