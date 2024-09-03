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
import sys
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ml-1m',choices=['ml-1m', 'ml-20m'])
parser.add_argument('--seed', default=3407, type=int)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--config_dir', default='./configs/')
parser.add_argument('--log_dir', default='./logs/')
parser.add_argument('--inference_only', default=False, type=bool)
args = parser.parse_args()
### 0. train baseline [optional]
### 1. split user in groups   [popularity, temprature]
### 2. train lora on metrics  [popularity, temprature]
### 3. train moe on metrics   [popularity, temprature]
### 4. Majority Vote / MoE on [popularity, temprature]
SKIP_SPLIT = True
TRAIN_LORA = False
TRAIN_MOE = True

if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ### -1. load dataset
    config = OmegaConf.load(f'{args.config_dir}{args.dataset}.yaml')
    print(f'initializing dataset: {args.dataset}')
    u2i_index, i2u_index = build_index(args.dataset)
    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    checkpoint_base_model = SASRec(usernum, itemnum, config).to(args.device)
    checkpoint_base_model.load_state_dict(torch.load(f'data/{args.dataset}/baseline/base.pth', map_location=torch.device(args.device)), strict=True)
    ## 0. train & eval baseline
    checkpoint_base_model.eval()
    with torch.no_grad():
        ndcg, ht = evaluate(checkpoint_base_model, dataset, config)
    print(f'base model ndcg: {ndcg}, ht: {ht}')
    del checkpoint_base_model
    ### 2. debug.


    model = SASRec(usernum, itemnum, config).to(args.device)
    model.load_state_dict(torch.load(f'data/{args.dataset}/baseline/base.pth', map_location=torch.device(args.device)), strict=True)
    lora_config = LoraConfig(
        r=32,
        lora_alpha=4,
        target_modules=[
            'attention_layers.0',
            'attention_layers.1'
        ],
        bias="none",
    )
    lora_model = get_peft_model(copy.deepcopy(model), lora_config)
    lora_model.eval()
    with torch.no_grad():
        ndcg, ht = evaluate(lora_model, dataset, config)
    print(f'base lora model ndcg: {ndcg}, ht: {ht}')

