import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_


class SumFusion(nn.Module):
    def __init__(self, args):
        super(SumFusion, self).__init__()
        
        self.fc_x = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.fc_y = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.layer_norm = nn.LayerNorm(args.embedding_dim, eps=1e-6)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, x, y):

        output = self.fc_x(self.layer_norm((x))) + self.fc_y(self.layer_norm((y)))

        return output

class ConcatFusion(nn.Module):
    def __init__(self,args):
        super(ConcatFusion, self).__init__()
        self.fc_out = nn.Linear(args.embedding_dim * 2, args.embedding_dim)
        self.layer_norm = nn.LayerNorm(args.embedding_dim, eps=1e-6)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, x, y):
        output = torch.cat((self.layer_norm(x), self.layer_norm(y)), dim=1)
        output = self.fc_out(output)
        return output


class FiLM(nn.Module):
    """
    FiLM: Visual Reasoning with a General Conditioning Layer,
    https://arxiv.org/pdf/1709.07871.pdf.
    """

    def __init__(self, args, x_film=True):
        super(FiLM, self).__init__()

        self.dim = args.embedding_dim
        self.fc = nn.Linear(args.embedding_dim, 2 * args.embedding_dim)
        self.fc_out = nn.Linear(args.embedding_dim, args.embedding_dim)

        self.x_film = x_film
                                                            
        self.layer_norm = nn.LayerNorm(args.embedding_dim, eps=1e-6)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, x, y):
 

        if self.x_film:
            film = self.layer_norm(x)
            to_be_film = self.layer_norm(y)
        else:
            film = self.layer_norm(y)
            to_be_film = self.layer_norm(x)

        gamma, beta = torch.split(self.fc(film), self.dim, 1)

        output = gamma * to_be_film + beta
        output = self.fc_out(output)

        return output

class GatedFusion(nn.Module):
    """
    Efficient Large-Scale Multi-Modal Classification,
    https://arxiv.org/pdf/1802.02892.pdf.
    """

    def __init__(self, args, x_gate=True):
        super(GatedFusion, self).__init__()

        self.fc_x = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.fc_y = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.fc_out = nn.Linear(args.embedding_dim, args.embedding_dim)

        self.x_gate = x_gate  # whether to choose the x to obtain the gate

        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(args.embedding_dim, eps=1e-6).to(args.local_rank)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, x, y):
                                                            
        out_x = self.fc_x(self.layer_norm(x))
        out_y = self.fc_y(self.layer_norm(y))

        if self.x_gate:
            gate = self.sigmoid(out_x)
            output = self.fc_out(torch.mul(gate, out_y))
        else:
            gate = self.sigmoid(out_y)
            output = self.fc_out(torch.mul(out_x, gate))

        return  output

class MLP(torch.nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.MLPlayer = nn.Sequential(
            # layer 1
            nn.Dropout(args.drop_rate, inplace=False),
            nn.Linear(args.embedding_dim, args.embedding_dim, bias=True),
            nn.ReLU(inplace=True),
            # layer 2
            nn.Dropout(args.drop_rate, inplace=False),
            nn.Linear(args.embedding_dim, args.embedding_dim, bias=True),
            nn.ReLU(inplace=True),
            # layer 3
            nn.Dropout(args.drop_rate, inplace=False),
            nn.Linear(args.embedding_dim, args.embedding_dim, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, input_embs):
        return self.MLPlayer(input_embs)


class SumDNN(nn.Module):
    def __init__(self, args):
        super(SumDNN, self).__init__()

        self.fc_x = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.fc_y = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.layer_norm = nn.LayerNorm(args.embedding_dim, eps=1e-6)

        self.dnn = MLP(args=args)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, x, y):
        output = self.fc_x(self.layer_norm((x))) + self.fc_y(self.layer_norm((y)))
        output = self.dnn(output)

        return output

class ConcatDNN(nn.Module):
    def __init__(self,args):
        super(ConcatDNN, self).__init__()
        self.fc_out = nn.Linear(args.embedding_dim * 2, args.embedding_dim)
        self.layer_norm = nn.LayerNorm(args.embedding_dim, eps=1e-6)
        self.dnn = MLP(args=args)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)


    def forward(self, x, y):
        output = torch.cat((self.layer_norm(x), self.layer_norm(y)), dim=1)
        output = self.dnn(self.fc_out(output))
        return output


class FiLMDNN(nn.Module):
    """
    FiLM: Visual Reasoning with a General Conditioning Layer,
    https://arxiv.org/pdf/1709.07871.pdf.
    """

    def __init__(self, args, x_film=True):
        super(FiLMDNN, self).__init__()

        self.dim = args.embedding_dim
        self.fc = nn.Linear(args.embedding_dim, 2 * args.embedding_dim)
        self.fc_out = nn.Linear(args.embedding_dim, args.embedding_dim)

        self.x_film = x_film

        self.layer_norm = nn.LayerNorm(args.embedding_dim, eps=1e-6)
        self.dnn = MLP(args=args)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, x, y):

        if self.x_film:
            film = self.layer_norm(x)
            to_be_film = self.layer_norm(y)
        else:
            film = self.layer_norm(y)
            to_be_film = self.layer_norm(x)

        gamma, beta = torch.split(self.fc(film), self.dim, 1)

        output = gamma * to_be_film + beta
        output = self.dnn(self.fc_out(output))

        return output


class GatedDNN(nn.Module):
    """
    Efficient Large-Scale Multi-Modal Classification,
    https://arxiv.org/pdf/1802.02892.pdf.
    """

    def __init__(self, args, x_gate=True):
        super(GatedDNN, self).__init__()

        self.fc_x = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.fc_y = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.fc_out = nn.Linear(args.embedding_dim, args.embedding_dim)

        self.x_gate = x_gate  # whether to choose the x to obtain the gate

        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(args.embedding_dim, eps=1e-6)

        self.dnn = MLP(args=args)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, x, y):

        out_x = self.fc_x(self.layer_norm(x))
        out_y = self.fc_y(self.layer_norm(y))

        if self.x_gate:
            gate = self.sigmoid(out_x)
            output = self.fc_out(torch.mul(gate, out_y))
        else:
            gate = self.sigmoid(out_y)
            output = self.fc_out(torch.mul(out_x, gate))

        output = self.dnn(output)

        return output
