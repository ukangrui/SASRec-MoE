import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
from models.sasrec_base import SASRec
import json
from utils import *
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
import optuna

DATASET = 'ml-1m'

num_u, num_i = get_usr_itm_num(DATASET)
train, test = load_train_test_data_num(load_txt_file(DATASET), num_i)
split_file_dir = 'config/popularity.json'
config = OmegaConf.load('config/lora.yaml')
with open(split_file_dir) as f:
    split_file = json.load(f)
all_splits = list(set(split_file.values()))

def objective(trial):
    r = trial.suggest_int("rank", 1, 16)
    alpha = trial.suggest_int("alpha", 4, 64)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-3)

    improv_list = []
    for split in all_splits:
        # Prepare data loaders
        lora_train, lora_test = get_lora_train_test_ds(split_file, train, test, split=split)
        train_loader = DataLoader(lora_train, batch_size=256, shuffle=True, collate_fn=collate_train)
        test_loader = DataLoader(lora_test, batch_size=256, shuffle=False, collate_fn=collate_test)

        # Load the base model and LoRA configuration
        base_model = SASRec(user_num=num_u ,item_num=num_i, maxlen=200, num_blocks=2, num_heads=1, hidden_units=50, dropout_rate=0.2)
        base_model.load_state_dict(torch.load('checkpoints/ml-1m-base.pth', map_location=torch.device('cuda')), strict=True)
        
        lora_config = LoraConfig(r=r, lora_alpha=alpha, target_modules=list(config.lora.target_modules), bias="none")
        model = get_peft_model(base_model, lora_config).to('cuda')

        # Define loss and optimizer
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        best_ndcg = -np.inf

        # Evaluate the base model
        base_ndcg = 0
        model.eval()
        with torch.no_grad():
            for test_batch in test_loader:
                u, seq, pos, test_items, mask = test_batch
                batch_ndcg, _ = eval_step(model, u, seq, pos, test_items, mask)
                base_ndcg += batch_ndcg
        base_ndcg /= len(lora_test)

        # Train and evaluate
        for epoch in range(25):
            model.train()
            for train_batch in train_loader:
                u, seq, pos, neg = train_batch
                train_step(model, u, seq, pos, neg, criterion, optimizer)

            model.eval()
            ndcg = 0
            with torch.no_grad():
                for test_batch in test_loader:
                    u, seq, pos, test_items, mask = test_batch
                    batch_ndcg, _ = eval_step(model, u, seq, pos, test_items, mask)
                    ndcg += batch_ndcg
            ndcg /= len(lora_test)

            if ndcg > best_ndcg:
                best_ndcg = ndcg

        improv = max(0, (best_ndcg - base_ndcg) / base_ndcg * 100)
        improv_list.append(improv)

    # Return the mean improvement as the objective value
    return np.mean(improv_list)

# Perform hyperparameter search
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Print best parameters
print("Best parameters:", study.best_params)
print("Best mean improvement (metric):", study.best_value)