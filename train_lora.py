import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
from models.sasrec_base import SASRec
import json
from utils import *
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model, PeftModel
import wandb

DATASET = 'ml-1m'

num_u, num_i = get_usr_itm_num(DATASET)
train, test = load_train_test_data_num(load_txt_file(DATASET), num_i)
split_file_dir = 'config/popularity.json'
with open(split_file_dir) as f:
    split_file = json.load(f)
all_splits = list(set(split_file.values()))


improv_list = []
for split in all_splits:
    print(f'training on {split}')

    # wandb.init(project="moelora",name=f"train_lora-{split}")

    lora_train, lora_test = get_lora_train_test_ds(split_file, train, test, split = split)
    train_loader = DataLoader(lora_train, batch_size = 256, shuffle = True, collate_fn = collate_train)
    test_loader  = DataLoader(lora_test, batch_size = 256, shuffle = False, collate_fn = collate_test)

    base_model = SASRec(user_num = num_u, item_num = num_i, maxlen = 200, num_blocks = 2, num_heads = 1, hidden_units = 50, dropout_rate = 0.2)
    base_model.load_state_dict(torch.load('checkpoints/ml-1m-base.pth', map_location=torch.device('cuda')), strict = True)

    config = OmegaConf.load('config/lora.yaml')
    lora_config = LoraConfig(
    r=int(config.lora.lora_rank),
    lora_alpha=float(config.lora.lora_alpha),
    target_modules=list(config.lora.target_modules),
    bias="none",

)

    model = get_peft_model(base_model, lora_config)
    model = model.to('cuda')

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    best_ndcg = -np.inf

    base_ndcg = 0
    model.eval()
    with torch.no_grad():
        for test_batch in test_loader:
            u, seq, pos, test_items, mask = test_batch
            batch_ndcg, batch_ht = eval_step(model, u, seq, pos, test_items, mask)
            base_ndcg += batch_ndcg
    base_ndcg /= len(lora_test)
    model.save_pretrained(f'lora_checkpoints/{split}/ndcg={round(base_ndcg,4)}')   

    
    for epoch in range(25):
        running_loss = 0.0
        model.train()
        for train_batch in train_loader:
            u, seq, pos, neg = train_batch
            running_loss += train_step(model, u, seq, pos, neg, criterion, optimizer)
        # print('Epoch', epoch, 'Loss', round(running_loss/len(lora_train),4), end=' ')
        # wandb.log({"epoch": epoch, "train-loss": round(running_loss/len(lora_train),4)})
        
        model.eval()
        ndcg, ht = 0, 0
        with torch.no_grad():
            for test_batch in test_loader:
                u, seq, pos, test_items, mask = test_batch
                batch_ndcg, batch_ht = eval_step(model, u, seq, pos, test_items, mask)
                ndcg += batch_ndcg
                ht += batch_ht
        # print(f'ndcg: {ndcg / len(lora_test)}, ht: { ht / len(lora_test)}')
        # wandb.log({"epoch": epoch, "ndcg": float(ndcg / len(lora_test))})
        # wandb.log({"epoch": epoch, "ht": float(ht / len(lora_test))})
        ndcg /= len(lora_test)
        if (ndcg > best_ndcg):
            best_ndcg = ndcg
            model.save_pretrained(f'lora_checkpoints/{split}/ndcg={round(best_ndcg,4)}')   
    
    if best_ndcg < base_ndcg:
        improv = 0
    else:
        improv = ((best_ndcg - base_ndcg)/base_ndcg) * 100
    
    improv_list.append(improv)
    # wandb.run.summary["improv"] = improv
    # wandb.finish()

print(np.mean(improv_list))