# 

import torch
from torch import nn

# taken from common.py in segment anythign facebook library 
class LayerNorm2d(nn.Module): 
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    



"""
We test different versions of decoders: 

Size of output of image encoder is 256x64x64

Excecpted output of the decoder is 1024x1024x3

"""


class Conv_Decoder(nn.Module):
    def __init__(self, p_dropout = 0.2):
        super().__init__()
        self._dropout1 = nn.Dropout(p=p_dropout)
        self._dropout2 = nn.Dropout(p=p_dropout)
        self._l_norm1 = LayerNorm2d(128)
        self._l_norm2 = LayerNorm2d(64)
        self._l_norm3 = LayerNorm2d(32)
        self._l_norm4 = LayerNorm2d(16)
        self._conv1 = nn.ConvTranspose2d(256, 128, kernel_size=2,stride=2, padding=0)
        self._conv2 = nn.ConvTranspose2d(128, 64, kernel_size=2,stride=2, padding=0)
        self._conv3 = nn.ConvTranspose2d(64, 32, kernel_size=2,stride=2, padding=0)
        self._conv4 = nn.ConvTranspose2d(32, 16, kernel_size=2,stride=2, padding=0)
        self._conv5 = nn.ConvTranspose2d(16, 1, kernel_size=1,stride=1, padding=0)

        self.decoder = nn.Sequential(
            self._conv1,
            self._l_norm1,
            nn.ReLU(),
            self._dropout1,
            self._conv2,
            self._l_norm2,
            nn.ReLU(),
            self._conv3,
            self._l_norm3,
            nn.ReLU(),
            self._dropout2,
            self._conv4,
            self._l_norm4,
            nn.ReLU(),
            self._conv5
            #, removed for numerical stability reasons 
            #nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)
    


class EncoderSAM_CustomDecoder(nn.Module):
    def __init__(self, sam_preprocess, sam_encoder, decoder, encoder_finetune_num_last_layers=5):
        super().__init__()
        self.sam_preprocess = sam_preprocess
        self.sam_encoder = sam_encoder
        num_layers_sam_encoder = 176

        last_layer_numb = 0
        for layer_number, param in enumerate(self.sam_encoder.parameters()):
            param.requires_grad = False
            last_layer_numb = layer_number
        print(f"Last layer number: {last_layer_numb}")

        # Unfreeze last layers of the encoder
        for layer_number, param in enumerate(self.sam_encoder.parameters()):
            if layer_number > last_layer_numb - encoder_finetune_num_last_layers:
                param.requires_grad = True
        
        # Unfreeze neck of the encoder
        self.sam_encoder.neck.requires_grad = True


        self.decoder = decoder

    def forward(self, x):
        x = self.sam_preprocess(x)
        x = self.sam_encoder(x)
        x = self.decoder(x)
        return x