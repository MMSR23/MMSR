import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision as tv
import torchvision.transforms as transforms
import pickle
import os
import random
from PIL import Image
import lmdb
import torch.distributed as dist
import math



class Build_text_CV_Dataset(Dataset):
    def __init__(self, u2seq, item_content, max_seq_len, item_num, text_size, db_path, item_id_to_keys,args):
        self.u2seq = u2seq
        self.item_content = item_content
        self.max_seq_len =  max_seq_len + 1 #修正
        self.item_num = item_num
        self.text_size = text_size
        self.db_path = db_path
        self.item_id_to_keys = item_id_to_keys
        self.args = args
        self.resize = args.CV_resize

        self.transform = transforms.Compose([
            tv.transforms.Resize((self.resize, self.resize)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])


    def __len__(self):
        return len(self.u2seq)


    def worker_init_fn(self, worker_id):
        initial_seed = torch.initial_seed() % 2 ** 31
        worker_seed = initial_seed + worker_id + self.args.local_rank + 8 * self.args.node_rank 
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    def __getitem__(self, index):
        seq = self.u2seq[index]
        seq_Len = len(seq) 
        tokens = seq[:-1] 
        tokens_Len = len(tokens) 
        mask_len_head = self.max_seq_len - seq_Len 
        log_mask = [0] * mask_len_head + [1] * tokens_Len 

        sample_items_text = np.zeros((2, self.max_seq_len, self.text_size * 2)) 
        sample_items_cv = np.zeros((2, self.max_seq_len, 3, self.resize, self.resize))

        # generate negative sample
        sam_neg_list = []
        for i in range(tokens_Len): 
            sam_neg = random.randint(1, self.item_num)
            while sam_neg in seq:
                sam_neg = random.randint(1, self.item_num)
            sam_neg_list.append(sam_neg)

        for i in range(tokens_Len):
            # pos
            sample_items_text[0][mask_len_head + i] = self.item_content[seq[i]] 
            # neg
            sample_items_text[1][mask_len_head + i] = self.item_content[sam_neg_list[i]]
        # target
        sample_items_text[0][mask_len_head + tokens_Len] = self.item_content[seq[-1]] 
        sample_items_text = torch.FloatTensor(sample_items_text).transpose(0, 1)

        env = lmdb.open(self.db_path, subdir=os.path.isdir(self.db_path),
                        readonly=True, lock=False,
                        readahead=False, meminit=False)
        with env.begin() as txn:
            for i in range(tokens_Len):
                # pos
                IMAGE = pickle.loads(txn.get(self.item_id_to_keys[seq[i]]))
                image_trans = self.transform(Image.fromarray(IMAGE.get_image()).convert('RGB'))
                sample_items_cv[0][mask_len_head + i] = image_trans
                # neg
                byteflow = txn.get(self.item_id_to_keys[sam_neg_list[i]])
                IMAGE = pickle.loads(byteflow)
                sample_items_cv[1][mask_len_head + i] = self.transform(Image.fromarray(IMAGE.get_image()).convert('RGB'))
            # target
            IMAGE = pickle.loads(txn.get(self.item_id_to_keys[seq[-1]]))
            image_trans = self.transform(Image.fromarray(IMAGE.get_image()).convert('RGB'))
            sample_items_cv[0][mask_len_head + tokens_Len] = image_trans
        sample_items_cv = torch.FloatTensor(sample_items_cv).transpose(0, 1)

        return sample_items_text, sample_items_cv, torch.FloatTensor(log_mask)


class LMDB_Image:
    def __init__(self, image, id):
        self.channels = image.shape[2]
        self.size = image.shape[:2]
        self.image = image.tobytes()
        self.id = id

    def get_image(self):
        image = np.frombuffer(self.image, dtype=np.uint8)
        return image.reshape(*self.size, self.channels)

class Build_Lmdb_Dataset(Dataset):
    def __init__(self, u2seq, item_num, max_seq_len, db_path, item_id_to_keys,args):

        self.u2seq = u2seq
        self.item_num = item_num
        self.max_seq_len = max_seq_len + 1 #修正
        self.db_path = db_path
        self.item_id_to_keys = item_id_to_keys
        self.resize = args.CV_resize

        self.transform = transforms.Compose([
            tv.transforms.Resize((self.resize, self.resize)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.args = args


    def __len__(self):
        return len(self.u2seq)
    
    def worker_init_fn(self, worker_id):
        
        initial_seed = torch.initial_seed() % 2 ** 31
        worker_seed = initial_seed + worker_id + self.args.local_rank + 8 * self.args.node_rank 
        random.seed(worker_seed)
        np.random.seed(worker_seed)


    def __getitem__(self, user_id):
        seq = self.u2seq[user_id]
        seq_Len = len(seq) 
        tokens_Len = len(seq) - 1
        mask_len_head = self.max_seq_len - seq_Len 
        log_mask = [0] * mask_len_head + [1] * tokens_Len

        sample_items = np.zeros((2, self.max_seq_len, 3, self.resize, self.resize))

        env = lmdb.open(self.db_path, subdir=os.path.isdir(self.db_path),
             readonly=True, lock=False,
             readahead=False, meminit=False)
        with env.begin() as txn:
            for i in range(tokens_Len):
                # pos
                IMAGE = pickle.loads(txn.get(self.item_id_to_keys[seq[i]]))
                image_trans = self.transform(Image.fromarray(IMAGE.get_image()).convert('RGB'))
                # image_trans = self.transform(Image.fromarray(IMAGE.get_image()))
                # clip-transform-aug
                # image_trans = self.transform(Image.fromarray(IMAGE.get_image()).convert('RGBA'))
                sample_items[0][mask_len_head + i] = image_trans
                # neg
                sam_neg = random.randint(1, self.item_num)
                while sam_neg in seq:
                    sam_neg = random.randint(1, self.item_num)
                byteflow = txn.get(self.item_id_to_keys[sam_neg])
                IMAGE = pickle.loads(byteflow)
                sample_items[1][mask_len_head + i] = self.transform(Image.fromarray(IMAGE.get_image()).convert('RGB'))
            # target
            IMAGE = pickle.loads(txn.get(self.item_id_to_keys[seq[-1]]))
            image_trans = self.transform(Image.fromarray(IMAGE.get_image()).convert('RGB'))
            sample_items[0][mask_len_head + tokens_Len] = image_trans
        sample_items = torch.FloatTensor(sample_items).transpose(0, 1)
        return sample_items, torch.FloatTensor(log_mask)

class Build_Text_Dataset(Dataset):
    def __init__(self, userseq, item_content, max_seq_len, item_num, text_size,args):
        self.userseq = userseq
        self.item_content = item_content
        self.max_seq_len =  max_seq_len + 1 #修正
        self.item_num = item_num
        self.text_size = text_size
        self.args = args

    def __len__(self):
        return len(self.userseq)

    def worker_init_fn(self, worker_id):
        initial_seed = torch.initial_seed() % 2 ** 31
        worker_seed = initial_seed + worker_id + self.args.local_rank + 8 * self.args.node_rank 
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        

    def __getitem__(self, index):
        seq = self.userseq[index]
        seq_Len = len(seq)
        tokens = seq[:-1]
        tokens_Len = len(tokens)
        mask_len_head = self.max_seq_len - seq_Len
        log_mask = [0] * mask_len_head + [1] * tokens_Len
        
        sample_items = np.zeros((2, self.max_seq_len, self.text_size * 2))

        for i in range(tokens_Len):
            # pos
            sample_items[0][mask_len_head + i] = self.item_content[seq[i]]
            # neg
            sam_neg = random.randint(1, self.item_num)
            while sam_neg in seq:
                sam_neg = random.randint(1, self.item_num)
            sample_items[1][mask_len_head + i] = self.item_content[sam_neg]
        # target
        sample_items[0][mask_len_head + tokens_Len] = self.item_content[seq[-1]]
        sample_items = torch.FloatTensor(sample_items).transpose(0, 1)
        return sample_items, torch.FloatTensor(log_mask)


class Build_Id_Dataset(Dataset):
    def __init__(self, u2seq, item_num, max_seq_len, args):
        self.u2seq = u2seq
        self.item_num = item_num
        self.max_seq_len = max_seq_len + 1 #修正
        self.args = args

    def __len__(self):
        return len(self.u2seq)

    def worker_init_fn(self, worker_id):
        initial_seed = torch.initial_seed() % 2 ** 31
        worker_seed = initial_seed + worker_id + self.args.local_rank + 8 * self.args.node_rank 
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    def __getitem__(self, user_id):
        seq = self.u2seq[user_id]
        seq_Len = len(seq)
        tokens_Len = seq_Len - 1
        mask_len_head = self.max_seq_len - seq_Len
        log_mask = [0] * mask_len_head + [1] * tokens_Len

        sample_items = []
        padding_seq = [0] * mask_len_head + seq
        sample_items.append(padding_seq)
        neg_items = []
        for i in range(tokens_Len):
            sam_neg = random.randint(1, self.item_num)
            while sam_neg in seq:
                sam_neg = random.randint(1, self.item_num)
            neg_items.append(sam_neg)
        neg_items = [0] * mask_len_head + neg_items + [0]
        sample_items.append(neg_items)
        sample_items = torch.LongTensor(np.array(sample_items)).transpose(0, 1)

        return sample_items, torch.FloatTensor(log_mask)

    
class Build_Id_Eval_Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]
    

class Build_Text_Eval_Dataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]

class Build_Lmdb_Eval_Dataset(Dataset):
    def __init__(self, data, item_id_to_keys, db_path, resize):
        self.data = data
        self.item_id_to_keys = item_id_to_keys
        self.db_path = db_path
        self.resize = resize
        self.padding_emb = Image.fromarray(np.zeros((224, 224, 3)).astype('uint8')).convert('RGB')

        self.transform = transforms.Compose([
                tv.transforms.Resize((self.resize, self.resize)),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        item_id = self.data[index]
        if index == 0:
            return self.transform(self.padding_emb)
        # env = self.env
        env = lmdb.open(self.db_path, subdir=os.path.isdir(self.db_path),
             readonly=True, lock=False,
             readahead=False, meminit=False)
        with env.begin() as txn:
            byteflow = txn.get(self.item_id_to_keys[item_id])
        IMAGE = pickle.loads(byteflow)
        img = self.transform(Image.fromarray(IMAGE.get_image()).convert('RGB'))
        return torch.FloatTensor(img)


# 同时输入text和CV的dataset
class Build_MMEncoder_Eval_Dataset(Dataset):
    def __init__(self, data_text, data_cv, item_id_to_keys, db_path, resize):
        self.data_cv = data_cv
        self.data_text = data_text
        self.item_id_to_keys = item_id_to_keys
        self.db_path = db_path
        self.resize = resize
        self.padding_emb = Image.fromarray(np.zeros((224, 224, 3)).astype('uint8')).convert('RGB')

        self.transform = transforms.Compose([
                tv.transforms.Resize((self.resize, self.resize)),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

    def __len__(self):
        return self.data_cv.shape[0] # or self.data_text.shape[0]

    def __getitem__(self, index):

        #text
        text = self.data_text[index]
        # cv
        item_id_cv = self.data_cv[index]
        if index == 0:
            return torch.LongTensor(text), self.transform(self.padding_emb)
        env = lmdb.open(self.db_path, subdir=os.path.isdir(self.db_path),
             readonly=True, lock=False,
             readahead=False, meminit=False)
        with env.begin() as txn:
            byteflow = txn.get(self.item_id_to_keys[item_id_cv])
        IMAGE = pickle.loads(byteflow)
        img = self.transform(Image.fromarray(IMAGE.get_image()).convert('RGB'))

        return torch.LongTensor(text), torch.FloatTensor(img)

# prepare for user encoder
class BuildEvalDataset(Dataset):

    def __init__(self, u2seq, item_content, max_seq_len, item_num):
        self.u2seq = u2seq
        self.item_content = item_content
        self.max_seq_len = max_seq_len + 1 
        self.item_num = item_num

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, user_id):

        seq = self.u2seq[user_id]
        tokens = seq[:-1] 
        target = seq[-1] 
        mask_len = self.max_seq_len - len(seq) 
        pad_tokens = [0] * mask_len + tokens
        log_mask = [0] * mask_len + [1] * len(tokens)
        input_embs = self.item_content[pad_tokens]
        labels = np.zeros(self.item_num)
        labels[target - 1] = 1.0
        return torch.LongTensor([user_id]), \
            input_embs, \
            torch.FloatTensor(log_mask), \
            labels


class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples