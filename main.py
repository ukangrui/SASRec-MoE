import argparse
import torch
import random
import numpy as np
from utils.utils import *
from utils.base_utils import *
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model, PeftModel
from models.model_lora import SASRec
from models.model_moe import SASRecMoE
from models.model_ensemble import SASRecEnsemble
import os
import logging
from datetime import datetime
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ml-1m-debug',choices=['ml-1m', 'ml-20m', 'ml-1m-debug'])
parser.add_argument('--seed', default=3407, type=int)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--config_dir', default='./configs/')
parser.add_argument('--log_dir', default='./logs/')
parser.add_argument('--inference_only', default=False, type=bool)
args = parser.parse_args()
### 0. train baseline [optional]
### 1. split user in groups   [popularity, temprature]
### 2. train lora on metrics  [popularity, temprature]
### 3. train moe on metrics   [popularity, temprature]
### 4. Majority Vote / MoE on [popularity, temprature]

if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'logs/training_{timestamp}.log'
    logging.basicConfig(filename=log_filename, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ### -1. load dataset
    config = OmegaConf.load(f'{args.config_dir}{args.dataset}.yaml')

    u2i_index, i2u_index = build_index(args.dataset)
    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    if args.inference_only:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        pass
    ### 0. trian baseline
    ### 1. split user groups
    for metric in config.metrics:
        split_users(dataset=args.dataset, mode=metric) ### split user in groups
    ### 2. train lora on groups
    for metric in config.metrics:
        base_metric = {}
        lora_metric = {}

        logging.info(f'training lora on {metric}')

        with open(f'data/{args.dataset}/lora/{metric}/{metric}.json', 'r') as json_file:
            user_groups = json.load(json_file)
        allsplits = sorted(np.unique(np.array(list(user_groups.values()))).tolist(), key=lambda x: int(x.split('_')[1]))

        model = SASRec(usernum, itemnum, config).to(args.device)
        model.load_state_dict(torch.load(f'data/{args.dataset}/baseline/base.pth', map_location=torch.device(args.device)), strict=True)

        num_batch = (len(user_train) - 1) // config.batch_size + 1
        sampler = WarpSampler(user_train, usernum, itemnum, batch_size=config.batch_size, maxlen=config.maxlen, n_workers=4)
        ### loop for each group split
        for train_split in allsplits:
            lora_config = LoraConfig(
                r=int(config.lora.lora_rank),
                lora_alpha=float(config.lora.lora_alpha),
                target_modules=list(config.lora.target_modules),
                bias="none",
            )
            lora_model = get_peft_model(copy.deepcopy(model), lora_config)
            with torch.no_grad():
                lora_model.eval()
                base_ndcg, base_ht = evaluate_split(model, dataset, config, user_groups, split=train_split)
                base_metric[train_split] = base_ndcg
            if float(base_ndcg) > config.skip_threshold: ### skip well trained split
                lora_model.save_pretrained(f'data/{str(args.dataset)}/lora/{str(metric)}/{str(train_split)}/ndcg={round(base_ndcg,3)}')
                lora_metric[train_split] = base_ndcg
                continue
            else:
                for name, param in lora_model.named_parameters():
                    if 'lora' not in name:
                        param.requires_grad = False
                lora_model.train()
                bce_criterion = torch.nn.BCEWithLogitsLoss()
                optimizer = torch.optim.Adam(lora_model.parameters(), lr=config.lora.lr, weight_decay=config.lora.weight_decay)
                best_ndcg = base_ndcg
                lora_model.save_pretrained(f'data/{str(args.dataset)}/lora/{str(metric)}/{str(train_split)}/ndcg={round(best_ndcg,3)}')
                #print("base best ndcg: ", best_ndcg)
                for epoch in range(config.lora.num_epochs):
                    lora_model.train()
                    for step in range(num_batch): 
                        u, seq, pos, neg = sampler.next_batch() 
                        u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)

                        ### train only on split
                        types_array = np.array([user_groups[str(user_id)] for user_id in u])
                        indices = np.where(types_array == str(train_split))[0]
                        u ,seq, pos, neg = u[indices], seq[indices], pos[indices], neg[indices]
                        pos_logits, neg_logits = lora_model(u, seq, pos, neg)
                        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device, requires_grad=True), torch.zeros(neg_logits.shape, device=args.device, requires_grad=True)
                        optimizer.zero_grad()
                        indices = np.where(pos != 0)
                        loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                        loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                        loss.backward()
                        optimizer.step()

                    lora_model.eval()
                    with torch.no_grad():
                        ndcg, ht = evaluate_split(lora_model, dataset, config, user_groups, split=train_split)

                    if (ndcg > best_ndcg):
                        best_ndcg = ndcg
                        lora_metric[train_split] = best_ndcg
                        lora_model.save_pretrained(f'data/{args.dataset}/lora/{metric}/{train_split}/ndcg={round(best_ndcg,3)}')
        logging.info(f'base ndcg: {np.mean(list(base_metric.values()))}')
        logging.info(f'lora ndcg: {np.mean(list(lora_metric.values()))}')
        sampler.close()
        clear_lora_cache()
    del model
    del sampler
    ## 3. train MoE
    for metric in config.metrics:
        logging.info(f'training MoE on {metric}')
        with open(f'data/{args.dataset}/lora/{metric}/{metric}.json', 'r') as json_file:
            user_groups = json.load(json_file)
        allsplits = sorted(np.unique(np.array(list(user_groups.values()))).tolist(), key=lambda x: int(x.split('_')[1])) ### split 1 -- 100
        num_batch = (len(user_train) - 1) // config.batch_size + 1
        sampler = WarpSampler(user_train, usernum, itemnum, batch_size=config.batch_size, maxlen=config.maxlen, n_workers=4)
        _model = SASRec(usernum, itemnum, config).to(args.device)
        _model.load_state_dict(torch.load(f'data/{args.dataset}/baseline/base.pth', map_location=torch.device(args.device)), strict=True)
        _ckpts = []
        for split in allsplits:
            split_dir = f'data/{args.dataset}/lora/{metric}/{split}'
            max_ndcg = max([float(i.split('=')[1]) for i in os.listdir(split_dir)])
            _ckpts.append(f'data/{args.dataset}/lora/{metric}/{split}/ndcg={max_ndcg}')
        attention_0_experts = [] 
        attention_1_experts = []
        for ckpt in _ckpts:
            lora_test_model = PeftModel.from_pretrained(copy.deepcopy(_model), ckpt)
            merged = lora_test_model.merge_and_unload()
            attention_0_experts.append(merged.attention_layers[0])
            attention_1_experts.append(merged.attention_layers[1])
            del lora_test_model
            del merged
        moe_model = SASRecMoE(usernum, itemnum, hard_gates=False,num_experts=config.num_groups, args=config, user_groups=user_groups).to(args.device)
        moe_model.load_state_dict(torch.load(f'data/{args.dataset}/baseline/base.pth', map_location=torch.device(args.device)), strict=False)
        moe_model.attention_layers = torch.nn.ModuleList(
            [
                torch.nn.ModuleList(attention_0_experts),
                torch.nn.ModuleList(attention_1_experts),
            ]
        )
        mark_only_gating_trainable(moe_model)
        bce_criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(moe_model.parameters(), lr=config.moe.lr, weight_decay=config.moe.weight_decay)
        best_ndcg = - np.inf
        for epoch in range(config.moe.num_epochs):
            moe_model.train()
            for step in range(num_batch): 
                u, seq, pos, neg = sampler.next_batch() 
                u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
                pos_logits, neg_logits, aux_loss = moe_model(u, seq, pos, neg)
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device, requires_grad=True), torch.zeros(neg_logits.shape, device=args.device, requires_grad=True)
                optimizer.zero_grad()
                indices = np.where(pos != 0)
                loss = bce_criterion(pos_logits[indices], pos_labels[indices]) + bce_criterion(neg_logits[indices], neg_labels[indices])
                print(loss,aux_loss)
                loss += aux_loss
                loss.backward()
                optimizer.step()
            
            moe_model.eval()
            with torch.no_grad():
                ndcg, ht = evaluate(moe_model, dataset, config)
                if ndcg > best_ndcg:
                    best_ndcg = ndcg
                    torch.save(moe_model.state_dict(),f'data/{args.dataset}/checkpoint/{metric}_ndcg={round(best_ndcg,3)}.pt')
        logging.info(f'best_ndcg: {best_ndcg}')
    ### 4. Majority Vote on MoE
    checkpoints = [
        "data/" + args.dataset + '/checkpoint/' + metric + '_ndcg=' +  round(max([float(i.split('=')[1]) for i in os.listdir(f"data/{args.dataset}/checkpoint/") if metric in i],3) + '.pt')
        for metric in config.metrics
    ]
    models = [
        torch.load(model_path, map_location='cpu')
        for model_path in checkpoints
    ]
    ndcg, ht = evaluate_ensemble(models, dataset, config, alpha= config.ensemble.alpha,mode='MoE')