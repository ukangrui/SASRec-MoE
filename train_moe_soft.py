import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
from models.sasrec_base import SASRec
import json
from utils import *
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model, PeftModel
import argparse
from models.sasrec_moe_soft import SASRecMoESoft
import os
from models.sasrec_base import SASRec
import pickle
parser = argparse.ArgumentParser()

parser.add_argument('--grouping_metric', type=str, default='temperature', choices=['popularity', 'temperature'])

parser.add_argument("--dataset", type=str, default="ml-1m")
parser.add_argument("--max_len", type=int, default=200)
parser.add_argument("--num_holdout", type=int, default=1)
parser.add_argument("--topK", type=int, default=10)
parser.add_argument('--device', type=str, default='mps')

args = parser.parse_args()

print(args.grouping_metric)

num_u, num_i = get_usr_itm_num(args.dataset)
# train, test = load_train_test_data_num(load_txt_file(args.dataset), num_i, max_len = args.max_len, num_test=args.num_holdout)
train = pickle.load(open("data/train_holdout4.pkl", "rb"))
test = pickle.load(open("data/test_holdout4.pkl", "rb"))
with open(f'config/{args.grouping_metric}.json') as f:
    split_file = json.load(f)
all_splits = list(set(split_file.values()))
train_loader = DataLoader(train, batch_size = 256, shuffle = True, collate_fn = collate_train)
test_loader  = DataLoader(test, batch_size = 256, shuffle = False, collate_fn = collate_test)

_ckpts = []
lora_checkpoint_ndcgs = []
for split in all_splits:
    split_dir = f'lora_checkpoints/{args.grouping_metric}/{split}'
    max_ndcg = max([float(i.split('=')[1]) for i in os.listdir(split_dir)])
    lora_checkpoint_ndcgs.append(max_ndcg)
    _ckpts.append(f'lora_checkpoints/{args.grouping_metric}/{split}/ndcg={max_ndcg}')
attention_0_experts = [] 
attention_1_experts = []

for ckpt in _ckpts:
    foo_model = SASRec(user_num = num_u, item_num = num_i, maxlen = 200, num_blocks = 2, num_heads = 1, hidden_units = 50, dropout_rate = 0.2, device = args.device)
    foo_model.load_state_dict(torch.load(f'checkpoints/{args.dataset}-base.pth', map_location=torch.device(args.device)))
    lora_checkpoint_model = PeftModel.from_pretrained(foo_model, ckpt).merge_and_unload()
    attention_0_experts.append(lora_checkpoint_model.attention_layers[0])
    attention_1_experts.append(lora_checkpoint_model.attention_layers[1])
    del lora_checkpoint_model
    del foo_model

moe_model_soft = SASRecMoESoft(user_num = num_u, item_num = num_i, maxlen = 200, num_blocks = 2, num_heads = 1, hidden_units = 50, dropout_rate = 0.2, device = args.device)
moe_model_soft.load_state_dict(torch.load(f'checkpoints/{args.dataset}-base.pth', map_location=torch.device(args.device)), strict=False)
moe_model_soft.attention_layers = torch.nn.ModuleList(
    [
        torch.nn.ModuleList(attention_0_experts),
        torch.nn.ModuleList(attention_1_experts),
    ]
)
moe_model_soft = moe_model_soft.to(args.device)

##
moe_model_soft.router = pretrain_router_w_distribution(moe_model_soft, moe_model_soft.router, train_loader, grouping_file=split_file, device=args.device)
##

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(moe_model_soft.parameters(), lr=1e-4)
for epoch in range(10):
    running_loss = 0.0
    moe_model_soft.train()
    for train_batch in train_loader:
        u, seq, pos, neg = train_batch
        running_loss += train_step(moe_model_soft, u, seq, pos, neg, criterion, optimizer, args.device)
    print('Epoch', epoch, 'Loss', running_loss * 100/len(train), end=' ')
    
    moe_model_soft.eval()
    ndcg, ht = 0, 0
    with torch.no_grad():
        for test_batch in test_loader:
            u, seq, pos, test_items, mask = test_batch
            batch_ndcg, batch_ht = eval_step(moe_model_soft, u, seq, pos, test_items, mask, topK = args.topK)
            ndcg += batch_ndcg
            ht += batch_ht
    print(f'ndcg: {ndcg / len(test)}, ht: { ht / len(test)}')


