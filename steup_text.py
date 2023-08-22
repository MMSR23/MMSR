import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
root_data_dir = os.path.abspath(os.path.join(BASE_DIR,".."))


dataset = 'hm'
behaviors = 'hm_pick_users_20W.tsv'
texts = 'hm_pick_items_20W.tsv'
lmdb_data = 'hm_image_20W.lmdb'

logging_num = 4
testing_num = 1

CV_resize = 224


# CV_model_load_list  = ['vit_mae_base']
# CV_model_load_list  = ["beit-base-224-16"]
# CV_model_load_list  = ["DeiTWithTeacher-224-16" ]
# CV_model_load_list  = ['DeiTBase-224-16' ]
# CV_model_load_list  = ['DeiTBase-224-16' ]
# CV_model_load_list  = ["googlevit-base-patch16-224"]
# CV_model_load_list  = ["openaiclip-vit-base-patch32"]
# CV_model_load_list  = ["SwinBase-224-7" ]
# CV_model_load_list  = ["resnet101.pth", 'resnet50.pth' ]

CV_model_load = None

BERT_model_load =  "RoBERTa_en"

# BERT_model_load = "chinese_bert-wwm-ext"
# BERT_model_load = "hflchinese-roberta-wwm-ext"  
# BERT_model_load =  "AlbertForMaskedLM"
# BERT_model_load =  "deberta_base"
# BERT_model_load =  "ElectraForPreTraining"
# BERT_model_load = "XLNetLMHeadModel"

CV_freeze_paras_before = 0
text_freeze_paras_before =  164 # Only two top transformer blocks

CV_fine_tune_lr = 0 
text_fine_tune_lr = 1e-4 


mode = 'train' # train test
item_tower = 'text-only' # modal, text, CV, ID

epoch = 150
load_ckpt_name = 'None' # From Scratch

l2_weight_list = [0.1]
drop_rate_list = [0.1]
batch_size_list = [64]
lr_list = [1e-4]
embedding_dim_list = [768]
max_seq_len_list = [20]

# fushion_list = ['sum', 'concat', 'film', 'gated','sum_dnn', 'concat_dnn', 'film_dnn','gated_dnn','co_att', 'merge_attn']
fusion = None

benchmark_list = ['sasrec'] # 'sasrec', 'grurec', 'nextit'

scheduler_steps = 120

for weight_decay in l2_weight_list:
    for batch_size in batch_size_list:
        for drop_rate in drop_rate_list:
            for embedding_dim in embedding_dim_list:
                for lr in lr_list:
                    for max_seq_len in max_seq_len_list:
                            for benchmark in benchmark_list:
                                label_screen = '{}_bs{}_ed{}_lr{}_dp{}_L2{}_len{}'.format(
                                        item_tower, batch_size, embedding_dim, lr,
                                        drop_rate, weight_decay, max_seq_len)


                                run_py = "CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' \
                                        python  -m torch.distributed.launch --nproc_per_node 8 --master_port 1251  run.py\
                                        --root_data_dir {}  --dataset {} --behaviors {} --texts {}  --lmdb_data {}\
                                        --mode {} --item_tower {} --load_ckpt_name {} --label_screen {} --logging_num {} --testing_num {}\
                                        --weight_decay {} --drop_rate {} --batch_size {} --lr {} --embedding_dim {}\
                                        --CV_resize {} --CV_model_load {} --bert_model_load {}  --epoch {} \
                                        --text_freeze_paras_before {} --CV_freeze_paras_before {} --max_seq_len {} \
                                        --CV_fine_tune_lr {} --text_fine_tune_lr {} --fusion_method {} --benchmark {} --scheduler_steps {}".format(
                                    root_data_dir, dataset, behaviors, texts, lmdb_data,
                                    mode, item_tower, load_ckpt_name, label_screen, logging_num, testing_num,
                                    weight_decay, drop_rate, batch_size, lr, embedding_dim,
                                    CV_resize, CV_model_load, BERT_model_load, epoch,
                                    text_freeze_paras_before, CV_freeze_paras_before, max_seq_len,
                                    CV_fine_tune_lr, text_fine_tune_lr, fusion, benchmark, scheduler_steps)

                                os.system(run_py)




