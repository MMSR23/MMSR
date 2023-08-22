
import torch
from torch import nn
from torch.nn.init import xavier_normal_
import torch.nn.functional as F

from .text_encoders import TextEmbedding

from .user_encoders import NextItNet, GRU4Rec, User_Encoder_SASRec
from .img_encoders import Vit_Encoder, Resnet_Encoder,VitOnlyEmbEncoder, ClipRN50Encoders

from .fushion_module import SumFusion, ConcatFusion, FiLM, GatedFusion, SumDNN, ConcatDNN,FiLMDNN,GatedDNN

from .modeling import CoAttention, MergedAttention

import numpy as np

class Model(torch.nn.Module):
    def __init__(self, args, item_num, bert_model, image_net):
        super(Model, self).__init__()
        self.args = args
        self.max_seq_len = args.max_seq_len + 1 #修正
        
        # various benchmark
        if "sasrec" in args.benchmark:
            self.user_encoder = User_Encoder_SASRec(
                item_num=item_num,
                max_seq_len=args.max_seq_len,
                item_dim=args.embedding_dim,
                num_attention_heads=args.num_attention_heads,
                dropout=args.drop_rate,
                n_layers=args.transformer_block)
        elif "nextit" in args.benchmark:
            self.user_encoder = NextItNet(args=args)
        elif "grurec"  in args.benchmark:
            self.user_encoder = GRU4Rec(args=args)

        # various encoders
        if "CV-only" in args.item_tower or "modal" in args.item_tower:
            if 'resnet' in args.CV_model_load:
                self.cv_encoder = Resnet_Encoder(image_net=image_net, args=args)
            elif "RN50" in args.CV_model_load:
                self.cv_encoder = ClipRN50Encoders(image_net=image_net, args=args)
            elif 'openaiclip-vit-base-patch32' in args.CV_model_load:
                self.cv_encoder = Vit_Encoder(image_net=image_net, args=args)
            elif "vit_mae_base" in args.CV_model_load:
                self.cv_encoder = Vit_Encoder(image_net=image_net, args=args)
            elif "googlevit-base-patch16-224" in args.CV_model_load:
                self.cv_encoder = Vit_Encoder(image_net=image_net, args=args)
            elif "Swin-tiny-224-7" in args.CV_model_load or "SwinBase-224-7"  in args.CV_model_load:
                self.cv_encoder = Vit_Encoder(image_net=image_net, args=args)

        if "text-only" in args.item_tower or "modal" in args.item_tower:
            self.text_encoder = TextEmbedding(args=args, bert_model=bert_model)

        if  "ID" in args.item_tower:
            self.id_encoder = nn.Embedding(item_num + 1, args.embedding_dim, padding_idx=0)
            xavier_normal_(self.id_encoder.weight.data)

        if "ID" not in args.item_tower and "CV-only" not in args.item_tower and "text-only" not in args.item_tower:
            # various fusion methods

            if args.fusion_method == 'sum':
                self.fusion_module = SumFusion(args=args)
            elif args.fusion_method == 'concat':
                self.fusion_module = ConcatFusion(args=args)
            elif args.fusion_method == 'film':
                self.fusion_module = FiLM(args=args, x_film=True)
            elif args.fusion_method == 'gated':
                self.fusion_module = GatedFusion(args=args, x_gate=True)
            elif args.fusion_method == 'sum_dnn':
                self.fusion_module = SumDNN(args=args)
            elif args.fusion_method == 'concat_dnn':
                self.fusion_module = ConcatDNN(args=args)
            elif args.fusion_method == 'film_dnn':
                self.fusion_module = FiLMDNN(args=args, x_film=True)
            elif args.fusion_method == 'gated_dnn':
                self.fusion_module = GatedDNN(args=args, x_gate=True)
            elif args.fusion_method == 'co_att' :
                self.fusion_module = CoAttention.from_pretrained("bert-base-uncased", args=args)
            elif args.fusion_method == 'merge_attn':
                self.fusion_module = MergedAttention.from_pretrained("bert-base-uncased",args=args)

        # loss
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, sample_items_id, sample_items_text, sample_items_CV, log_mask, local_rank, args):
        if "modal" in args.item_tower:
            # text mask
            batch_size, num_words = sample_items_text.shape
            num_words = num_words // 2
            text_mask = torch.narrow(sample_items_text, 1, num_words, num_words)
            # text and img last hidden states
            hidden_states_text = self.text_encoder(sample_items_text.long())
            hidden_states_CV = self.cv_encoder(sample_items_CV)

            if args.fusion_method in ['sum', 'concat', 'film', 'gated','sum_dnn', 'concat_dnn', 'film_dnn','gated_dnn']:
                text_mask_expanded = text_mask.unsqueeze(-1).expand(hidden_states_text.size()).float()
                hidden_states_text = torch.sum(hidden_states_text * text_mask_expanded, 1) / torch.clamp(text_mask_expanded.sum(1), min=1e-9)
                if 'resnet' in args.CV_model_load or 'RN50' in args.CV_model_load:
                    hidden_states_CV = hidden_states_CV  # resnet
                else:
                    hidden_states_CV = torch.mean(hidden_states_CV, dim=1)  # mean
                input_embs = self.fusion_module(hidden_states_text, hidden_states_CV)
            if args.fusion_method in ['co_att', 'merge_attn']:
                CV_mask = torch.ones(hidden_states_CV.size()[0], hidden_states_CV.size()[1]).to(local_rank)
                input_embs = self.fusion_module(hidden_states_text, text_mask, hidden_states_CV, CV_mask,local_rank)

        if "text-only" in args.item_tower:
            batch_size, num_words = sample_items_text.shape
            num_words = num_words // 2
            text_mask = torch.narrow(sample_items_text, 1, num_words, num_words)
            hidden_states_text = self.text_encoder(sample_items_text.long())
            text_mask_expanded = text_mask.unsqueeze(-1).expand(hidden_states_text.size()).float().to(local_rank)       
            input_embs = torch.sum(hidden_states_text * text_mask_expanded, 1) / torch.clamp(text_mask_expanded.sum(1), min=1e-9) # mean

        if "CV-only" in args.item_tower:
            hidden_states_CV = self.cv_encoder(sample_items_CV)

            if 'resnet' in args.CV_model_load or 'RN50' in args.CV_model_load:
                input_embs = hidden_states_CV  # resnet
            else:
                input_embs = torch.mean(hidden_states_CV, dim=1)  

        if "ID" in args.item_tower:
            input_embs = self.id_encoder(sample_items_id)

        input_embs = input_embs.view(-1, self.max_seq_len, 2, self.args.embedding_dim)


        pos_items_embs = input_embs[:, :, 0]
        neg_items_embs = input_embs[:, :, 1]

        input_logs_embs = pos_items_embs[:, :-1, :]
        target_pos_embs = pos_items_embs[:, 1:, :]
        target_neg_embs = neg_items_embs[:, :-1, :]

        # various benchmark
        if "sasrec" in args.benchmark:
             prec_vec = self.user_encoder(input_logs_embs, log_mask, local_rank)
        elif "nextit" in args.benchmark:
            prec_vec = self.user_encoder(input_logs_embs)
        elif "grurec"  in args.benchmark:
            prec_vec = self.user_encoder(input_logs_embs)

        pos_score = (prec_vec * target_pos_embs).sum(-1)
        neg_score = (prec_vec * target_neg_embs).sum(-1)
        pos_labels, neg_labels = torch.ones(pos_score.shape).to(local_rank), torch.zeros(neg_score.shape).to(local_rank)

        indices = torch.where(log_mask != 0)

        loss_1 = self.criterion(pos_score[indices], pos_labels[indices]) 
        loss_2 = self.criterion(neg_score[indices], neg_labels[indices])
        loss = loss_1 + loss_2 

        return loss
