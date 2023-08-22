
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import os
from statistics import mode
import torch
import time
import numpy as np
import random
from pathlib import Path
import re

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

import torch.optim as optim
from parameters import parse_args

from model.model import Model

from data_utils import *
from data_utils.tools import read_texts_bert, read_behaviors_text, get_doc_input_bert,read_images,read_behaviors_CV
from data_utils.utils import para_and_log, report_time_train, report_time_eval, save_model,setuplogger,get_time
from data_utils.dataset import Build_text_CV_Dataset, Build_Lmdb_Dataset,Build_Text_Dataset,Build_Id_Dataset
from data_utils.metrics import get_text_only_scoring, get_text_fusion_scoring, get_itemId_scoring,\
                                get_LMDB_only_scoring, get_LMDB_fusion_scoring,get_MMEncoder_scoring
from data_utils.lr_decay import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup,get_step_schedule_with_warmup


import torchvision.models as models
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, \
                        ElectraConfig, ElectraTokenizer, ElectraModel,\
                        XLNetConfig, XLNetModel,XLNetTokenizer
from transformers import AutoConfig, AutoTokenizer, AutoModel
import clip


from transformers import CLIPVisionModel, ViTMAEModel, BeitModel,DeiTModel,ViTModel, SwinModel


os.environ["TOKENIZERS_PARALLELISM"] = "false"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_eval_all(now_epoch, max_epoch, early_stop_epoch, max_eval_value, early_stop_count,
                    model, user_history, users_eval, batch_size, item_num, 
                    mode, is_early_stop, local_rank,args, Log_file, 
                    item_content=None, item_id_to_keys=None):

    eval_start_time = time.time()
    if 'text' in args.item_tower:
        if 'text-only' in args.item_tower:
            Log_file.info('get text-only scoring...')
            item_scoring = get_text_only_scoring(model, item_content, batch_size, args, local_rank)
        else:
            Log_file.info('get text-fusion scoring...')
            item_scoring = get_text_fusion_scoring(model, item_content, batch_size, args, local_rank)

    if 'CV' in args.item_tower:
        if 'CV-only' in args.item_tower:
            Log_file.info('get CV-only scoring...')
            item_scoring = get_LMDB_only_scoring(model, item_num, item_id_to_keys, batch_size, args, local_rank)
        else:
            Log_file.info('get CV-fusion scoring ...')
            item_scoring = get_LMDB_fusion_scoring(model, item_num, item_id_to_keys, batch_size, args, local_rank)

    if 'modal' in args.item_tower:
        Log_file.info('get Multi-modal (text and CV) scoring...')
        item_scoring = get_MMEncoder_scoring(model, item_content, item_num, item_id_to_keys, batch_size, args, local_rank)

    elif "ID"  in args.item_tower:
        Log_file.info('get ID scoring...')
        item_scoring = get_itemId_scoring(model, item_num, batch_size, args, local_rank)

    valid_Hit10, nDCG10= eval_model(model, user_history, users_eval, item_scoring, 
                                    batch_size, args,item_num, Log_file, mode, local_rank)

    report_time_eval(eval_start_time, Log_file)
    Log_file.info('')
    need_break = False
    if valid_Hit10 > max_eval_value:
        max_eval_value = valid_Hit10
        max_epoch = now_epoch
        early_stop_count = 0
    else:
        early_stop_count += 1
        if early_stop_count > 20: 
            if is_early_stop:
                need_break = True
            early_stop_epoch = now_epoch
    return max_eval_value, max_epoch, early_stop_epoch, early_stop_count, need_break



def run_train_all(local_rank, model_dir,Log_file ,Log_screen, start_time, args):

    # ============================ text and image encoders============================

    if 'modal' in args.item_tower or 'text-only' in args.item_tower:

        if "bert-base-en" in args.bert_model_load:
            # bert
            Log_file.info('load bert model...')
            bert_model_load = os.path.abspath(os.path.join(BASE_DIR, "TextEncoders", args.bert_model_load))
            tokenizer = AutoTokenizer.from_pretrained(bert_model_load)
            config = AutoConfig.from_pretrained(bert_model_load, output_hidden_states=True)
            bert_model = AutoModel.from_pretrained(bert_model_load, config=config)
            pooler_para = [197, 198] 
            # If the last two layers are not used, they need to be frozen, 
            # otherwise the parameter has a gradient under DDP but is not used

        elif "RoBERTa_en" in args.bert_model_load:
            # RoBERTa_en
            Log_file.info('load RoBERTa model...')
            bert_model_load = os.path.abspath(os.path.join(BASE_DIR, "TextEncoders", args.bert_model_load))
            tokenizer = RobertaTokenizer.from_pretrained(bert_model_load)
            config = RobertaConfig.from_pretrained(bert_model_load, output_hidden_states=True)
            bert_model = RobertaModel.from_pretrained(bert_model_load, config=config)
            pooler_para = [197, 198] 
            # If the last two layers are not used, they need to be frozen, 
            # otherwise the parameter has a gradient under DDP but is not used

        elif "ElectraForPreTraining" in args.bert_model_load:
            # Electra
            Log_file.info('load Electra model...')
            bert_model_load = os.path.abspath(os.path.join(BASE_DIR, "TextEncoders", args.bert_model_load))
            tokenizer = ElectraTokenizer.from_pretrained(bert_model_load)
            config = ElectraConfig.from_pretrained(bert_model_load, output_hidden_states=True)
            bert_model = ElectraModel.from_pretrained(bert_model_load, config=config)
            pooler_para = []

        elif "XLNetLMHeadModel" in args.bert_model_load:
            # XLNet
            Log_file.info('load XLNet model...')
            bert_model_load = os.path.abspath(os.path.join(BASE_DIR, "TextEncoders", args.bert_model_load))
            tokenizer = XLNetTokenizer.from_pretrained(bert_model_load)
            config = XLNetConfig.from_pretrained(bert_model_load, output_hidden_states=True)
            bert_model = XLNetModel.from_pretrained(bert_model_load, config=config)
            pooler_para = []

        for index, (name, param) in enumerate(bert_model.named_parameters()):
            if index > args.text_freeze_paras_before:
                param.requires_grad = False

        if 'text-only' in args.item_tower:
            cv_model = None


    if 'modal' in args.item_tower or 'CV-only' in args.item_tower:
        if "vit_mae_base" in args.CV_model_load:
            Log_file.info('load MAE model...')
            cv_model_load = os.path.abspath(os.path.join(BASE_DIR, "CVEncoders", args.CV_model_load))
            # mae
            cv_model = ViTMAEModel.from_pretrained(cv_model_load)

        elif "beit-base-224-16" in args.CV_model_load:
            Log_file.info('load beit model...')
            cv_model_load = os.path.abspath(os.path.join(BASE_DIR, "CVEncoders", args.CV_model_load))
            # beit
            cv_model = BeitModel.from_pretrained(cv_model_load)

        elif "googlevit-base-patch16-224"in args.CV_model_load:
            Log_file.info('load vit model...')
            cv_model_load = os.path.abspath(os.path.join(BASE_DIR, "CVEncoders", args.CV_model_load))
            # vit
            cv_model = ViTModel.from_pretrained(cv_model_load)

        elif "Swin-tiny-224-7"in args.CV_model_load or "SwinBase-224-7" in args.CV_model_load:
            Log_file.info('load swin model...')
            cv_model_load = os.path.abspath(os.path.join(BASE_DIR, "CVEncoders", args.CV_model_load))
            # swin
            cv_model = SwinModel.from_pretrained(cv_model_load)

        elif "resnet" in args.CV_model_load:
            Log_file.info('load resnet model...')
            cv_model_load = os.path.abspath(os.path.join(BASE_DIR, "CVEncoders","resnet", args.CV_model_load))

            if '18' in cv_model_load:
                cv_model = models.resnet18(pretrained=False)
            elif '34' in cv_model_load:
                cv_model = models.resnet34(pretrained=False)
            elif '50' in cv_model_load:
                cv_model = models.resnet50(pretrained=False)
            elif '101' in cv_model_load:
                cv_model = models.resnet101(pretrained=False)
            elif '152' in cv_model_load:
                cv_model = models.resnet152(pretrained=False)
            else:
                cv_model = None
            cv_model.load_state_dict(torch.load(cv_model_load)) 


        for index, (name, param) in enumerate(cv_model.named_parameters()):
            if index > args.CV_freeze_paras_before:
                param.requires_grad = False

        if 'CV-only' in args.item_tower:
            bert_model = None

    if 'ID' in args.item_tower:
        bert_model = None
        cv_model = None

    
    # ============================ load data ============================
    # text can directly read content into memory because it takes up less memory
    item_content = None
    if 'modal' in args.item_tower or 'text' in args.item_tower:
        Log_file.info("Read Texts ...")
        item_dic_titles_after_tokenizer, before_item_name_to_index, before_item_index_to_name = read_texts_bert(
            os.path.join(args.root_data_dir, args.dataset, args.texts), args, tokenizer)

        Log_file.info("Read behavious ...")

        item_num, item_dic_titles_after_tokenizer, item_name_to_index, users_train, users_valid, users_test, users_history_for_valid, users_history_for_test = \
            read_behaviors_text(os.path.join(args.root_data_dir, args.dataset, args.behaviors),
                        item_dic_titles_after_tokenizer, before_item_name_to_index,
                        before_item_index_to_name, args.max_seq_len, args.min_seq_len, Log_file)

        Log_file.info('combine text information...')
        text_title, text_title_attmask = get_doc_input_bert(item_dic_titles_after_tokenizer, item_name_to_index, args)


        item_content = np.concatenate([x for x in [
            text_title,
            text_title_attmask
        ] if x is not None], axis=1)
    
    # image cannot directly read content into memory because it takes up a lot of memory
     # This work uses lmdb to store images separately, and batch reading can also be very fast
     # When processing user seq in the early stage, image and Id are processed in a similar way, so they are put together
    item_id_to_keys = None
    if 'modal' in args.item_tower or 'CV' in args.item_tower or 'ID' in args.item_tower:
        Log_file.info('read images...')

        before_item_name_to_id, before_item_id_to_keys = read_images(
            os.path.join(args.root_data_dir, args.dataset, args.texts))

        Log_file.info('read behaviors...')
        item_num, item_id_to_keys, users_train, users_valid, users_test, users_history_for_valid, users_history_for_test = \
            read_behaviors_CV(os.path.join(args.root_data_dir, args.dataset, args.behaviors), before_item_id_to_keys,
                            before_item_name_to_id, args.max_seq_len, args.min_seq_len, Log_file)
    
    print("users_train", len(users_train))

    # ============================ dataset and dataloader ============================
    if 'modal' in args.item_tower:
        Log_file.info('build  text and CV dataset...')
        train_dataset = Build_text_CV_Dataset(u2seq=users_train,
                                            item_content=item_content,
                                            max_seq_len=args.max_seq_len,
                                            item_num=item_num,
                                            text_size=args.num_words_title,
                                            db_path=os.path.join(args.root_data_dir, args.dataset, args.lmdb_data),
                                            item_id_to_keys=item_id_to_keys,
                                            args=args)
    elif 'CV' in args.item_tower:
        train_dataset = Build_Lmdb_Dataset(u2seq=users_train,
                                           item_num=item_num,
                                           max_seq_len=args.max_seq_len,
                                           db_path=os.path.join(args.root_data_dir, args.dataset, args.lmdb_data),
                                           item_id_to_keys=item_id_to_keys, 
                                           args=args)

    elif "text" in args.item_tower:
        train_dataset = Build_Text_Dataset(userseq=users_train, 
                                           item_content=item_content, 
                                           max_seq_len=args.max_seq_len,
                                           item_num=item_num, 
                                           text_size=args.num_words_title,
                                           args=args)

    elif "ID" in args.item_tower:
        train_dataset = Build_Id_Dataset(u2seq=users_train, 
                                         item_num=item_num, 
                                         max_seq_len=args.max_seq_len,
                                         args=args)


    Log_file.info('build DDP sampler...')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    Log_file.info('build dataloader...')
    train_dl = DataLoader(train_dataset, 
                        batch_size=args.batch_size, 
                        num_workers=args.num_workers,
                        multiprocessing_context="fork", 
                        worker_init_fn=train_dataset.worker_init_fn, 
                        pin_memory=True, 
                        sampler=train_sampler)

    # ============================ Model ============================
    Log_file.info('build model...')
    model = Model(args, item_num, bert_model, cv_model).to(local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)


    try:
        epoches = []
        for file in os.listdir(model_dir):
            epoches.append(int(re.split(r'[._-]', file)[1]))
        args.load_ckpt_name = 'epoch-' + str(max(epoches)) + '.pt'
        #############
        print("args.load_ckpt_name", args.load_ckpt_name)
        ckpt_path = os.path.abspath(os.path.join(model_dir, args.load_ckpt_name))
        start_epoch = int(re.split(r'[._-]', args.load_ckpt_name)[1])
        print("start_epoch", start_epoch)
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        Log_file.info('load checkpoint...')
        model.load_state_dict(checkpoint['model_state_dict'])
        Log_file.info(f"Model loaded from {args.load_ckpt_name}")
        torch.set_rng_state(checkpoint['rng_state']) 
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])  
        is_early_stop = False

    except Exception as e:
        checkpoint = None  # new
        ckpt_path = None  # new
        start_epoch = 0
        is_early_stop = False


    Log_file.info('model.cuda()...')
    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # ============================ OPtimizer ============================

    image_net_params = []
    bert_params = []
    recsys_params = []

    for index, (name, param) in enumerate(model.module.named_parameters()):
        if param.requires_grad:
            if 'cv_encoder' in name:
                if 'cv_proj' in name or "classifier" in name:  # cv_proj: clip_vit, mae ; classifier: swin_tiny, swin_base ; fc:resnet
                    recsys_params.append(param)
                elif "resnet" in name and "fc" in name:
                    recsys_params.append(param)
                elif "grid_encoder" in name:
                    recsys_params.append(param)
                else:
                    image_net_params.append(param)
            elif "text_encoder" in name:
                if "text_pooler" in name:
                    recsys_params.append(param)
                else:
                    bert_params.append(param)
            else:
                recsys_params.append(param)


    optimizer = optim.AdamW([
        {'params': bert_params,'lr': args.text_fine_tune_lr, 'weight_decay': 0,  'initial_lr': args.text_fine_tune_lr},
        {'params': image_net_params, 'lr': args.CV_fine_tune_lr, 'weight_decay': 0, 'initial_lr': args.CV_fine_tune_lr},
        {'params': recsys_params, 'lr': args.lr, 'weight_decay': args.weight_decay, 'initial_lr': args.lr}
        ])
    

    Log_file.info("***** {} finetuned parameters in text encoder *****".format(
        len(list(bert_params))))
    Log_file.info("***** {} fiuetuned parameters in image encoder*****".format(
        len(list(image_net_params))))
    Log_file.info("***** {} parameters with grad in recsys *****".format(
        len(list(recsys_params))))


    model_params_require_grad = []
    model_params_freeze = []
    for param_name, param_tensor in model.module.named_parameters():
        if param_tensor.requires_grad:
            model_params_require_grad.append(param_name)
        else:
            model_params_freeze.append(param_name)

    Log_file.info("***** model: {} parameters require grad, {} parameters freeze *****".format(
        len(model_params_require_grad), len(model_params_freeze)))


    
    if 'None' not in args.load_ckpt_name:  
        optimizer.load_state_dict(checkpoint["optimizer"])
        Log_file.info(f"optimizer loaded from {ckpt_path}")

    # ============================  Training ============================

    total_num = sum(p.numel() for p in model.module.parameters())
    trainable_num = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
    Log_file.info("##### total_num {} #####".format(total_num))
    Log_file.info("##### trainable_num {} #####".format(trainable_num))

    Log_file.info('\n')
    Log_file.info('Training...')
    next_set_start_time = time.time()
    max_epoch, early_stop_epoch = 0, args.epoch
    max_eval_value, early_stop_count = 0, 0

    steps_for_log, steps_for_eval = para_and_log(model, len(users_train), args.batch_size, Log_file,
                                                 logging_num=args.logging_num, testing_num=args.testing_num)
    Log_screen.info('{} train start'.format(args.label_screen))

    # mix
    from torch.cuda.amp import autocast as autocast
    scaler = torch.cuda.amp.GradScaler()

    if 'None' not in args.load_ckpt_name:  
        scaler.load_state_dict(checkpoint["scaler_state"])
        Log_file.info(f"scaler loaded from {ckpt_path}")


    # lr dacay without warmup
    warmup_steps = 0 
    if args.scheduler == "cosine_schedule":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=args.epoch,
            start_epoch=start_epoch)
        
    elif args.scheduler == "linear_schedule":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=args.epoch,
            start_epoch=start_epoch)
        
    elif args.scheduler == "step_schedule":
        lr_scheduler = get_step_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            gap_steps = args.scheduler_steps, # Decline by 10 times every N rounds
            start_epoch=start_epoch)
    else:
        lr_scheduler = None
        

    epoch_left = args.epoch - start_epoch
    for ep in range(epoch_left):
        now_epoch = start_epoch + ep + 1
        train_dl.sampler.set_epoch(now_epoch)
        Log_file.info('\n')
        Log_file.info('epoch {} start'.format(now_epoch))
        Log_file.info('')
        loss, batch_index, need_break = 0.0, 1, False
        model.train()

        if lr_scheduler is not None:
            Log_file.info('start of trainin epoch:  {} ,lr: {}'.format(now_epoch, lr_scheduler.get_lr()))

        loss = 0
        for data in train_dl:
            if 'modal' in args.item_tower:
                sample_items_text, sample_items_CV, log_mask = data
                sample_items_text, sample_items_CV,log_mask = sample_items_text.to(local_rank), \
                                                              sample_items_CV.to(local_rank),log_mask.to(local_rank)
                sample_items_text = sample_items_text.view(-1, args.num_words_title * 2)
                sample_items_CV = sample_items_CV.view(-1, 3, args.CV_resize, args.CV_resize)
                sample_items_id = None

            elif  'text' in args.item_tower:
                sample_items_text, log_mask = data
                sample_items_text, log_mask = sample_items_text.to(local_rank), log_mask.to(local_rank)
                sample_items_text = sample_items_text.view(-1, args.num_words_title * 2)
                sample_items_CV = None
                sample_items_id = None

            elif 'CV' in args.item_tower:
                sample_items_CV, log_mask = data
                sample_items_CV, log_mask = sample_items_CV.to(local_rank), log_mask.to(local_rank)
                sample_items_CV =  sample_items_CV.view(-1, 3, args.CV_resize, args.CV_resize)
                sample_items_text = None
                sample_items_id = None

            elif "ID" in args.item_tower:
                sample_items, log_mask = data
                sample_items, log_mask = sample_items.to(local_rank), log_mask.to(local_rank)
                sample_items_id = sample_items.view(-1)
                sample_items_text = None
                sample_items_CV = None

            else:
                pass

            optimizer.zero_grad()


            # mix
            with autocast(enabled=True):
                bz_loss = model(sample_items_id,sample_items_text, sample_items_CV, log_mask, local_rank, args)
                loss += bz_loss.data.float()

            scaler.scale(bz_loss).backward()
            scaler.step(optimizer)
            scaler.update()


            if torch.isnan(loss.data):
                need_break = True
                break

            if batch_index % steps_for_log == 0:
                Log_file.info('cnt: {}, Ed: {}, batch loss: {:.5f}, sum loss of modal: {:.5f}'.format(
                    batch_index, batch_index * args.batch_size, loss.data / batch_index, loss.data))
            batch_index += 1
            
        if not need_break and now_epoch % 5 == 0:
            # valid
            max_eval_value, max_epoch, early_stop_epoch, early_stop_count, need_break = \
                run_eval_all(now_epoch, max_epoch, early_stop_epoch, max_eval_value, early_stop_count,
                        model, users_history_for_valid, users_valid, 128, item_num, 
                        args.mode, is_early_stop, local_rank, args, Log_file, item_content,item_id_to_keys)
            # test
            run_eval_all(now_epoch, max_epoch, early_stop_epoch, max_eval_value, early_stop_count,
                        model, users_history_for_test, users_test, 128, item_num,
                        args.mode, is_early_stop, local_rank, args, Log_file, item_content,item_id_to_keys)

            model.train()

        # 存每个epoch的权重
        if dist.get_rank() == 0:
            save_model(now_epoch, model, model_dir, optimizer, torch.get_rng_state(), torch.cuda.get_rng_state(), Log_file)   # full

        if lr_scheduler is not None:
            lr_scheduler.step()
            Log_file.info('end of trainin epoch:  {} ,lr: {}'.format(now_epoch, lr_scheduler.get_lr()))

        Log_file.info('')
        next_set_start_time = report_time_train(batch_index, now_epoch, loss, next_set_start_time, start_time, Log_file)
        Log_file.info('{} training: epoch {}/{}'.format(args.label_screen, now_epoch, args.epoch))

        if need_break:
            break


    Log_file.info('\n')
    Log_file.info('%' * 90)
    Log_file.info(' max eval Hit10 {:0.5f}  in epoch {}'.format(max_eval_value * 100, max_epoch))
    Log_file.info(' early stop in epoch {}'.format(early_stop_epoch))
    Log_file.info('the End')
    Log_screen.info('{} train end in epoch {}'.format(args.label_screen, early_stop_epoch))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False



def main():

    args = parse_args()
    local_rank = int(os.environ['RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    args.local_rank = local_rank 
    args.node_rank = 0

    setup_seed(123456) 

    #记录
    if 'modal' in args.item_tower:
        dir_label =  str(args.behaviors).strip().split(".")[0] + "_"  + str(args.item_tower)
        log_paras = f"bs{args.batch_size}" \
                    f"_ed_{args.embedding_dim}_lr_{args.lr}" \
                    f"_L2_{args.weight_decay}_dp_{args.drop_rate}_FlrText_{args.text_fine_tune_lr}_FlrImg_{args.CV_fine_tune_lr}"\
                    f"_{args.bert_model_load}_{args.CV_model_load}" \
                    f'_freeze_{args.text_freeze_paras_before}_{args.CV_freeze_paras_before}'\
                    f"_len_{args.max_seq_len}" \
                    f"_{args.fusion_method}"\
                    f"_{args.benchmark}"\
                    f"_{args.scheduler}{args.scheduler_steps}"


    elif  "text-only" in args.item_tower:
        dir_label =  str(args.behaviors).strip().split(".")[0] + "_"  + str(args.item_tower) 
        log_paras = f"bs{args.batch_size}" \
            f"_ed_{args.embedding_dim}_lr_{args.lr}" \
            f"_L2_{args.weight_decay}_dp_{args.drop_rate}_FlrText_{args.text_fine_tune_lr}"\
            f"_{args.bert_model_load}"\
            f'_freeze_{args.text_freeze_paras_before}'\
            f"_len_{args.max_seq_len}" \
            f"_{args.fusion_method}"\
            f"_{args.benchmark}"\
            f"_{args.scheduler}{args.scheduler_steps}"


    elif "CV-only" in args.item_tower:
        dir_label =  str(args.behaviors).strip().split(".")[0] + "_"  + str(args.item_tower)
        log_paras = f"bs{args.batch_size}" \
            f"_ed_{args.embedding_dim}_lr_{args.lr}" \
            f"_L2_{args.weight_decay}_dp_{args.drop_rate}_FlrImg_{args.CV_fine_tune_lr}"\
            f"_{args.CV_model_load}"\
            f'_freeze_{args.CV_freeze_paras_before}'\
            f"_len_{args.max_seq_len}"\
            f"_{args.fusion_method}"\
            f"_{args.benchmark}"\
            f"_{args.scheduler}{args.scheduler_steps}"

    elif "ID" in args.item_tower:
        dir_label =  str(args.behaviors).strip().split(".")[0] + "_"  + str(args.item_tower)
        log_paras = f"bs{args.batch_size}" \
            f"_ed_{args.embedding_dim}_lr_{args.lr}" \
            f"_L2_{args.weight_decay}_dp_{args.drop_rate}" \
            f"_len_{args.max_seq_len}"\
            f"_{args.benchmark}"\
            f"_{args.scheduler}{args.scheduler_steps}"


    model_dir = os.path.join("./checkpoint_" + dir_label, f"cpt_" + log_paras)
    time_run = time.strftime("%Y-%m-%d, %H:%M:%S", time.localtime())
    Log_file, Log_screen = setuplogger(dir_label, log_paras, time_run, args.mode, dist.get_rank())
    Log_file.info(args)
    Log_file.info(time_run) 

    if not os.path.exists(model_dir):
        Path(model_dir).mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    if "train" in args.mode:
        run_train_all(local_rank, model_dir,Log_file ,Log_screen, start_time, args)

    end_time = time.time()
    hour, minute, seconds = get_time(start_time, end_time)
    Log_file.info("#### (time) all: hours {} minutes {} seconds {} ####".format(hour, minute, seconds))


if __name__ == '__main__':
    main()



