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
import os
import logging
from datetime import datetime
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ml-1m',choices=['ml-1m', 'ml-20m'])
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
TRAIN_BASE = True
TRAIN_SPLIT = True
TRAIN_LORA = True
TRAIN_MOE = False
TRAIN_ENSEMBLE = False

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
    logging.info(f'lora_rank: {config.lora.lora_rank}, lora_alpha: {config.lora.lora_alpha}, lora_lr: {config.lora.lr}, modules: {config.lora.target_modules}, num groups: {config.num_groups}')
    print(f'initializing dataset: {args.dataset}')
    u2i_index, i2u_index = build_index(args.dataset)
    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    if args.inference_only:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        pass

    # 0. train & eval baseline
    if TRAIN_BASE:
        checkpoint_base_model = SASRec(usernum, itemnum, config).to(args.device)
        checkpoint_base_model.load_state_dict(torch.load(f'data/{args.dataset}/baseline/base.pth', map_location=torch.device(args.device)), strict=True)
        checkpoint_base_model.eval()
        with torch.no_grad():
            checkpoint_ndcg, checkpoint_ht = evaluate_batch(checkpoint_base_model, dataset, config)
        logging.info(f'base model ndcg: {checkpoint_ndcg}, ht: {checkpoint_ht}')
        print(f'base model ndcg: {checkpoint_ndcg}, ht: {checkpoint_ht}')
        del checkpoint_base_model

    ## 1. split user groups
    if TRAIN_SPLIT:
        print('building user groups')
        for metric in config.metrics:
            split_users(dataset=args.dataset, mode=metric, num_groups=config.num_groups) ### split user in groups

    ### 2. train lora on groups
    if TRAIN_LORA:
        dummy_flag = True
        for metric in config.metrics:
            base_metric_ndcg = {}
            lora_metric_ndcg = {}
            base_metric_ht = {}
            lora_metric_ht = {}

            logging.info(f'training lora on {metric}')
            print(f'training lora on {metric}')

            with open(f'data/{args.dataset}/lora/{metric}/num_groups={config.num_groups}/{metric}.json', 'r') as json_file:
                user_groups = json.load(json_file)
            allsplits = sorted(np.unique(np.array(list(user_groups.values()))).tolist(), key=lambda x: int(x.split('_')[1]))
            assert len(allsplits) == config.num_groups

            num_batch = (len(user_train) - 1) // config.batch_size + 1
            sampler = WarpSampler(user_train, usernum, itemnum, batch_size=config.batch_size, maxlen=config.maxlen, n_workers=4)
            ### loop for each group split
            for train_split in tqdm(allsplits):
                dummy_model = SASRec(usernum, itemnum, config).to(args.device)
                dummy_model.load_state_dict(torch.load(f'data/{args.dataset}/baseline/base.pth', map_location=torch.device(args.device)), strict=True)
                lora_config = LoraConfig(
                    r=int(config.lora.lora_rank),
                    lora_alpha=float(config.lora.lora_alpha),
                    target_modules=list(config.lora.target_modules),
                    bias="none",
                )
                lora_model = get_peft_model(dummy_model, lora_config)
                del dummy_model
                if dummy_flag:
                    lora_model.print_trainable_parameters()
                    dummy_flag = False
                with torch.no_grad():
                    lora_model.eval()
                    base_ndcg_lora, base_ht_lora = evaluate_split(lora_model, dataset, config, user_groups, split=train_split)
                    base_metric_ndcg[train_split] = base_ndcg_lora
                    base_metric_ht[train_split] = base_ht_lora
                lora_model.save_pretrained(f'data/{str(args.dataset)}/lora/{str(metric)}/num_groups={config.num_groups}/{str(train_split)}/ndcg={round(base_ndcg_lora,5)}')
                if float(base_ndcg_lora) > config.skip_threshold: ### skip well trained split
                    lora_metric_ndcg[train_split] = base_ndcg_lora
                    lora_metric_ht[train_split] = base_ht_lora
                    continue
                else:
                    lora_model.train()
                    bce_criterion = torch.nn.BCEWithLogitsLoss()
                    optimizer = torch.optim.Adam(lora_model.parameters(), lr=config.lora.lr, weight_decay=config.lora.weight_decay)
                    best_ndcg_lora = base_ndcg_lora
                    best_ht_lora = base_ht_lora
                    lora_metric_ndcg[train_split] = best_ndcg_lora
                    lora_metric_ht[train_split] = best_ht_lora
                    for epoch in range(config.lora.num_epochs):
                        lora_model.train()
                        for step in range(num_batch): 
                            u, seq, pos, neg = sampler.next_batch() 
                            ###
                            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
                            types_array = np.array([user_groups[str(user_id)] for user_id in u])
                            indices = np.where(types_array == str(train_split))[0]
                            u ,seq, pos, neg = u[indices], seq[indices], pos[indices], neg[indices]
                            ###
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
                            ndcg_lora, ht_lora = evaluate_split(lora_model, dataset, config, user_groups, split=train_split)

                        if (ndcg_lora > best_ndcg_lora):
                            best_ndcg_lora = ndcg_lora
                            best_ht_lora = ht_lora
                            lora_metric_ndcg[train_split] = best_ndcg_lora
                            lora_metric_ht[train_split]=  best_ht_lora
                            lora_model.save_pretrained(f'data/{args.dataset}/lora/{metric}/num_groups={config.num_groups}/{train_split}/ndcg={round(best_ndcg_lora,5)}')
                    del lora_model
            logging.info(f'base ndcg: {np.mean(list(base_metric_ndcg.values()))}, ht: {np.mean(list(base_metric_ht.values()))}')
            logging.info(f'lora ndcg: {np.mean(list(lora_metric_ndcg.values()))}, ht: {np.mean(list(lora_metric_ht.values()))}')
            print(f'base ndcg: {np.mean(list(base_metric_ndcg.values()))}, ht: {np.mean(list(base_metric_ht.values()))}')
            print(f'lora ndcg: {np.mean(list(lora_metric_ndcg.values()))}, ht: {np.mean(list(lora_metric_ht.values()))}')
            sampler.close()
            clear_lora_cache()
        del sampler

    ## 3. train MoE
    if TRAIN_MOE:
        for metric in config.metrics:
            logging.info(f'training MoE on {metric}')
            print(f'training MoE on {metric}')
            with open(f'data/{args.dataset}/lora/{metric}/num_groups={config.num_groups}/{metric}.json', 'r') as json_file:
                user_groups = json.load(json_file)
            allsplits = sorted(np.unique(np.array(list(user_groups.values()))).tolist(), key=lambda x: int(x.split('_')[1])) ### split 1 -- 100
            num_batch = (len(user_train) - 1) // config.batch_size + 1
            sampler = WarpSampler(user_train, usernum, itemnum, batch_size=config.batch_size, maxlen=config.maxlen, n_workers=4)
            _ckpts = []
            lora_checkpoint_ndcgs = []
            for split in allsplits:
                split_dir = f'data/{args.dataset}/lora/{metric}/num_groups={config.num_groups}/{split}'
                max_ndcg = max([float(i.split('=')[1]) for i in os.listdir(split_dir)])
                lora_checkpoint_ndcgs.append(max_ndcg)
                _ckpts.append(f'data/{args.dataset}/lora/{metric}/num_groups={config.num_groups}/{split}/ndcg={max_ndcg}')
            attention_0_experts = [] 
            attention_1_experts = []

            for ckpt in _ckpts:
                foo_model = SASRec(usernum, itemnum, config).to(args.device)
                foo_model.load_state_dict(torch.load(f'data/{args.dataset}/baseline/base.pth', map_location=torch.device(args.device)), strict=True)
                lora_checkpoint_model = PeftModel.from_pretrained(foo_model, ckpt).merge_and_unload()
                attention_0_experts.append(lora_checkpoint_model.attention_layers[0])
                attention_1_experts.append(lora_checkpoint_model.attention_layers[1])
                del lora_checkpoint_model
                del foo_model
            # Hard
            moe_model_hard = SASRecMoE(usernum, itemnum, hard_gates=True,num_experts=config.num_groups , user_groups=user_groups, gamma = 0, args=config,
                                       user_embs= torch.nn.Embedding(1,1)).to(args.device)
            moe_model_hard.load_state_dict(torch.load(f'data/{args.dataset}/baseline/base.pth', map_location=torch.device(args.device)), strict=False)
            moe_model_hard.attention_layers = torch.nn.ModuleList(
                [
                    torch.nn.ModuleList(attention_0_experts),
                    torch.nn.ModuleList(attention_1_experts),
                ]
            )
            moe_model_hard.eval()
            with torch.no_grad():
                ndcg_hard, ht_hard = evaluate_batch(moe_model_hard, dataset, config)
            logging.info(f'MoE Hard ndcg: {ndcg_hard}, ht: {ht_hard}')
            print(f'MoE Hard ndcg: {ndcg_hard}, ht: {ht_hard}')
            torch.save(moe_model_hard,f'data/{args.dataset}/checkpoint/hard_{metric}_ndcg={round(ndcg_hard,5)}.pt')
            del moe_model_hard
            # Soft
            moe_model_soft = SASRecMoE(usernum, itemnum, hard_gates=False,num_experts=config.num_groups, user_groups=user_groups, gamma = config.moe.gamma, args=config,
                                       user_embs=contruct_split_emb_from_item(dataset=args.dataset, group_data=user_groups)).to(args.device)
            moe_model_soft.load_state_dict(torch.load(f'data/{args.dataset}/baseline/base.pth', map_location=torch.device(args.device)), strict=False)
            moe_model_soft.attention_layers = torch.nn.ModuleList(
                [
                    torch.nn.ModuleList(attention_0_experts),
                    torch.nn.ModuleList(attention_1_experts),
                ]
            )
            mark_only_router_trainable(moe_model_soft)
            bce_criterion = torch.nn.BCEWithLogitsLoss()
            optimizer = torch.optim.SGD(moe_model_soft.parameters(), lr=config.moe.lr)
            best_ndcg_soft = - np.inf
            best_ht_soft = -np.inf
            for epoch in range(config.moe.num_epochs):
                moe_model_soft.train()
                running_loss = 0
                for step in range(num_batch): 
                    u, seq, pos, neg = sampler.next_batch() 
                    u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
                    pos_logits, neg_logits, usergate = moe_model_soft(u, seq, pos, neg)
                    pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device, requires_grad=True), torch.zeros(neg_logits.shape, device=args.device, requires_grad=True)
                    optimizer.zero_grad()
                    indices = np.where(pos != 0)
                    loss = bce_criterion(pos_logits[indices], pos_labels[indices]) + bce_criterion(neg_logits[indices], neg_labels[indices])
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.detach().item()
                print(f'Epoch: {epoch}, loss: {running_loss}')
                moe_model_soft.eval()
                with torch.no_grad():
                    ndcg_soft, ht_soft = evaluate_batch(moe_model_soft, dataset, config)
                    print(f'Epoch: {epoch}, ndcg: {ndcg_soft}')
                    if ndcg_soft > best_ndcg_soft:
                        best_ndcg_soft = ndcg_soft
                        best_ht_soft = ht_soft
                        torch.save(moe_model_soft,f'data/{args.dataset}/checkpoint/soft_{metric}_ndcg={round(best_ndcg_soft,5)}.pt')
            logging.info(f'MoE Soft ndcg: {best_ndcg_soft}, ht: {best_ht_soft}')
            print(f'MoE Soft ndcg: {best_ndcg_soft}, ht: {best_ht_soft}')

    ## 4. Majority Vote on MoE
    if TRAIN_ENSEMBLE:
        logging.info(f'Ensembling with alpha = {list(config.ensemble.alpha)}')
        ensemble_checkpoints = list()
        for metric in config.metrics:
            metric_checkpoints = sorted([i for i in os.listdir(f"data/{args.dataset}/checkpoint/") if metric in i], key=lambda x: float(x.split('=')[1].replace('.pt','')))[-1]
            ensemble_checkpoints.append(os.path.join('data',args.dataset, 'checkpoint', metric_checkpoints))
        models = [
            torch.load(model_path, map_location=args.device)
            for model_path in ensemble_checkpoints
        ]
        for model in models:
            model.eval()
        ndcg_ensemble, ht_ensemble = evaluate_ensemble(models, dataset, config, alpha= list(config.ensemble.alpha))
        logging.info(f'Ensemble ndcg: {ndcg_ensemble}, ht: {ht_ensemble}')