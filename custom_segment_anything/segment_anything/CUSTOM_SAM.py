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
    

class MLP_Decoder(nn.Module):

    # the input shape is: 

    # ([1, 256, 64, 64])


    # 1024 / 16 = 64
    # hence each of the 16x16 patches is now represented as a 256 dimensional vector, 
    # there are 64x64 of these patches: 


    def __init__(self, p_dropout = 0.2):
        super().__init__()
        # fill in here: 
        self.flatten = nn.Flatten(start_dim=2)
        self.dropout1 = nn.Dropout(p=p_dropout)
        self.dropout2 = nn.Dropout(p=p_dropout)
        self.dropout3 = nn.Dropout(p=p_dropout)
        self.dropout4 = nn.Dropout(p=p_dropout)
        self.dropout5 = nn.Dropout(p=p_dropout)

        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 256)

        self.gelu = nn.LeakyReLU()


    def forward(self, x):
        # input is of shape: dim_0 = batch_size,  ([1, 256, 64, 64])
        batch_size = x.shape[0]
        
        x = torch.swapaxes(x, 1, 2)
        x = torch.swapaxes(x, 2, 3)
        # x is now of shape: (1,64,64,256)
        # we want to apply a linear layer to each of the 4096 patches

        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.gelu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.gelu(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        x = self.gelu(x)
        x = self.dropout4(x)

        x = self.fc5(x)
        x = self.gelu(x)
        x = self.dropout5(x)

        x = self.fc6(x)

        x = x.reshape(batch_size, 64, 64, 16, 16)
        x = torch.swapaxes(x, 2, 3)
        # we want shape of x to be (batch_size, 1, 1024, 1024)
        x = x.reshape(batch_size, 1, 1024, 1024)
        return x
    
class SAM_Encoder_Custom_Decoder(nn.Module):
    def __init__(self, sam_preprocess, sam_encoder, decoder, encoder_finetune_num_last_layers=5):
        super().__init__()
        self.sam_preprocess = sam_preprocess
        self.sam_encoder = sam_encoder
    
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