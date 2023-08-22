import torch
import torch.nn as nn

from torch.nn.init import xavier_normal_, constant_


class Vit_Encoder(torch.nn.Module):
    def __init__(self, image_net, args):
        super(Vit_Encoder, self).__init__()

        self.image_net = image_net
        if "SwinBase-224-7"  in args.CV_model_load:
            self.cv_proj = nn.Linear(1024, args.embedding_dim)
        else:
            self.cv_proj = nn.Linear(args.word_embedding_dim, args.embedding_dim)
            
        xavier_normal_(self.cv_proj.weight.data)
        if self.cv_proj.bias is not None:
            constant_(self.cv_proj.bias.data, 0)

    def forward(self, item_content):
        
        last_hidden_state_CV = self.image_net(item_content)[0]
        last_hidden_state_CV = self.cv_proj(last_hidden_state_CV)
        return last_hidden_state_CV

    
class VitOnlyEmbEncoder(torch.nn.Module):
    def __init__(self, image_net, args):
        super(VitOnlyEmbEncoder, self).__init__()

        self.image_net = image_net
        self.cv_proj = nn.Linear(args.word_embedding_dim, args.embedding_dim)
        xavier_normal_(self.cv_proj.weight.data)
        if self.cv_proj.bias is not None:
            constant_(self.cv_proj.bias.data, 0)

    def forward(self, item_content):
        # get hidden_states 
        hidden_state_CV = self.image_net(item_content)[2]
        hidden_state_CV_emb = self.cv_proj(hidden_state_CV[0])

        return hidden_state_CV_emb
    
class ClipRN50Encoders(torch.nn.Module):
    def __init__(self, image_net, args):
        super(ClipRN50Encoders, self).__init__()

        self.resnet = image_net
        self.flag = False
        if 'CV-only' not in args.item_tower and args.fusion_method == 'co_att' or args.fusion_method == 'merge_attn':
            self.flag = True
            self.grid_encoder = nn.Sequential(
                nn.Conv2d(2048, 768, kernel_size=3, stride=1, padding=1, bias=False),
                # nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(inplace=True))
        else:
            self.cv_proj = nn.Linear(1024, args.embedding_dim)
            xavier_normal_(self.cv_proj.weight.data)
            if self.cv_proj.bias is not None:
                constant_(self.cv_proj.bias.data, 0)

    def forward(self, item_content):

        x = self.resnet(item_content)
        if not self.flag:
            x = self.cv_proj(x)
            return x

        x = self.grid_encoder(x)
        x = x.permute(0, 2, 3, 1)
        x = x.view(x.shape[0], -1, x.shape[-1])

        return x

class Resnet_Encoder(torch.nn.Module):
    def __init__(self, image_net,args):
        super(Resnet_Encoder, self).__init__()
        self.resnet = image_net

        self.flag = False
        if 'CV-only' not in args.item_tower and args.fusion_method == 'co_att' or args.fusion_method == 'merge_attn':
            self.flag = True
            self.grid_encoder = nn.Sequential(
                nn.Conv2d(2048, 768, kernel_size=3, stride=1, padding=1, bias=False),
                # nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
            )
        else:
            num_fc_ftr = self.resnet.fc.in_features
            self.resnet.fc = nn.Linear(num_fc_ftr, args.embedding_dim)

            xavier_normal_(self.resnet.fc.weight.data)
            if self.resnet.fc.bias is not None:
                constant_(self.resnet.fc.bias.data, 0)


    def forward(self, item_content):

        x = self.resnet(item_content)

        if not self.flag:
            return x
        x = self.grid_encoder(x)
        x = x.permute(0, 2, 3, 1)
        x = x.view(x.shape[0], -1, x.shape[-1])

        return x

