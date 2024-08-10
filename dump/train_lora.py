import torch
import argparse
from models.model_lora import SASRec
from utils.base_utils import *
import os
import pytorch_lightning as pl
from peft import LoraConfig, get_peft_model, PeftModel
import json

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ml-1m',)
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
    with open('checkpoints/ml1m_temprature.json', 'r') as json_file:
        user_types = json.load(json_file)
    alltypes = np.unique(np.array(list(user_types.values()))) ### usersplit group
    u2i_index, i2u_index = build_index(args.dataset)
    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    num_batch = (len(user_train) - 1) // args.batch_size + 1
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    print(usernum, itemnum)
    model = SASRec(usernum, itemnum, args).to(args.device)
    model.load_state_dict(torch.load('checkpoints/base.pth', map_location=torch.device(args.device)), strict=True)



    for train_split in alltypes:
        lora_config = LoraConfig(
            r=1,
            lora_alpha=64,
            target_modules=[
                'attention_layers.0',
                'attention_layers.1'
            ],
            bias="none",
        )

        lora_model = get_peft_model(model, lora_config)
        with torch.no_grad():
            lora_model.eval()
            base_ndcg, base_ht = evaluate_split(model, dataset, args, user_types_dic=user_types, split=train_split)

        if float(base_ndcg) > 0.125: ### skip well trained split
            lora_model.save_pretrained(f'lora_checkpoint_temp/{train_split}/ndcg={round(base_ndcg,3)}')
            print('skipping well trained usersplit: ', train_split)
            continue
        else:
            print('-----------------------------')
            print('tuning user_split = ', train_split)
            for name, param in lora_model.named_parameters():
                if 'lora' not in name:
                    param.requires_grad = False
            lora_model.train()
            bce_criterion = torch.nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(lora_model.parameters(), lr=args.lr, weight_decay=0.01)
            best_ndcg = base_ndcg
            lora_model.save_pretrained(f'lora_checkpoint_temp/{train_split}/ndcg={round(best_ndcg,3)}')
            print("base best ndcg: ", best_ndcg)
            for epoch in range(args.num_epochs):
                lora_model.train()
                for step in range(num_batch): 
                    u, seq, pos, neg = sampler.next_batch() 
                    u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)

                    ### train only on split
                    types_array = np.array([user_types[str(user_id)] for user_id in u])
                    indices = np.where(types_array == str(train_split))[0]
                    u ,seq, pos, neg = u[indices], seq[indices], pos[indices], neg[indices]
                    if u.size == 0:
                        print('Error')
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
                    ndcg, ht = evaluate_split(lora_model, dataset, args, user_types_dic=user_types, split=train_split)

                if (ndcg > best_ndcg):
                    best_ndcg = ndcg
                    lora_model.save_pretrained(f'lora_checkpoint_temp/{train_split}/ndcg={round(best_ndcg,3)}')
            #     print('current_loss: ', running_loss)
            print('updated_best_bdcg: ', best_ndcg)
            print('-----------------------------')
    sampler.close()


    # alltypes = sorted(np.unique(np.array(list(user_types.values()))).tolist(), key=lambda x: int(x.split('_')[1])) ### usersplit group
    # _ckpts = []
    # ndcg = {}
    # for type in alltypes:
    #     type_dir = 'lora_checkpoint_popu/' + type + '/'
    #     max_ndcg = max([float(i.split('=')[1]) for i in os.listdir(type_dir)])
    #     _ckpts.append('lora_checkpoint_popu/' + type + '/' + 'ndcg=' +str(max_ndcg))
    # for ckpt in _ckpts:
    #     print(ckpt)
    #     lora_test_model = PeftModel.from_pretrained(copy.deepcopy(model), ckpt)
    #     curr_id = ckpt.split('_')[3].split('/')[0]
    #     merged = lora_test_model.merge_and_unload()
    #     with torch.no_grad():
    #         merged.eval()
    #         base_ndcg, base_ht = evaluate_split(merged, dataset, args, user_types_dic=user_types, split='split_' + curr_id)
    #         ndcg[curr_id]  = base_ndcg
    
    # json_data = json.dumps(ndcg)
    # with open('lora_popularity.json', 'w') as json_file:
    #     json_file.write(json_data)
            