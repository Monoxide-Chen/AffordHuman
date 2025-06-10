import torch
import os
import argparse
import random
import numpy as np
import yaml
import torch.distributed as dist
from tools.trainer import train
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from dataset_utils.dataset_3DIR import _3DIR
from tools.models.model_LEMON_d import LEMON as LEMON_d
from tools.models.model_LEMON_p import LEMON as LEMON_p
from tools.models.model_LEMON_laso import LEMON as LEMON_laso
from tools.models.LEMON_noCur import LEMON_wocur
from tools.utils.logger import Logger

def main(opt, dict):
    if not os.path.exists(opt.save_checkpoint_path):
        os.makedirs(opt.save_checkpoint_path, exist_ok=True)

    if opt.use_gpu and dict['run_type']=='train':
        dist.init_process_group(backend='nccl', init_method='env://')
        rank = dist.get_rank()
        size = dist.get_world_size()
        local_rank = int(os.environ['LOCAL_RANK'])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        if opt.use_gpu:
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

    train_dataset = _3DIR(dict['train_image'], dict['train_pts'], dict['human_3DIR'],
                          dict['behave'], mode='train',
                          desc_file=dict.get('desc_file'))
    train_sampler = DistributedSampler(train_dataset)
    val_dataset = _3DIR(dict['val_image'], dict['val_pts'], dict['human_3DIR'],
                        dict['behave'], mode='val',
                        desc_file=dict.get('desc_file'))
    val_sampler = DistributedSampler(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, sampler=train_sampler, num_workers=8, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, sampler=val_sampler, num_workers=8)

    logger = Logger(os.path.join(opt.save_checkpoint_path, 'log.txt'), title="eval_matrix")
    logger.set_names(["Epoch", 'AUC', 'aIOU', 'SIM', 'Precision', 'Recall', 'F1', 'geo_fn', 'geo_fp','MSE'])

    model_type = dict.get('model', 'd')
    if model_type == 'laso':
        model = LEMON_laso(dict['emb_dim'], run_type='train', device=device)
    elif model_type == 'no_cur':
        model = LEMON_wocur(dict['emb_dim'], run_type='train', device=device)
    elif model_type == 'p':
        model = LEMON_p(dict['emb_dim'], run_type='train', device=device)
    else:
        model = LEMON_d(dict['emb_dim'], run_type='train', device=device)
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True, broadcast_buffers=False)
    train(opt, dict, train_loader, train_sampler, val_loader, val_dataset, model, logger, device, rank, model_type)

    logger.close()

def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def read_yaml(path):
    file = open(path, 'r', encoding='utf-8')
    string = file.read()
    dict = yaml.safe_load(string)
    return dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=12, help='batch_size')
    parser.add_argument('--save_checkpoint_path', type=str, default='runs/LEMON/', help='save_checkpoint_path')
    parser.add_argument('--yaml', type=str, default='config/train.yaml', help='train setting')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu to run')
    opt = parser.parse_args()
    dict = read_yaml(opt.yaml)
    seed_torch(42)
    main(opt, dict)
