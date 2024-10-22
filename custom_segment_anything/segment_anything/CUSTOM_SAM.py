# 

import torch
from torch import nn


class Conv_Decoder(nn.Module):

    # Source: https://www.kaggle.com/code/yogendrayatnalkar/promptless-taskspecific-finetuning-of-metaai-sam

    def __init__(self, p_dropout = 0.2):
        super().__init__()
        self.dropout1 = nn.Dropout(p=p_dropout)
        self.dropout2 = nn.Dropout(p=p_dropout)

        self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=2,stride=2, padding=0)
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=2,stride=2, padding=0)
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=2,stride=2, padding=0)
        self.conv4 = nn.ConvTranspose2d(32, 16, kernel_size=2,stride=2, padding=0)
        self.conv5 = nn.ConvTranspose2d(16, 1, kernel_size=1,stride=1, padding=0)

        self.lrelu = nn.ReLU()
        
    def forward(self, x):
        #return self.decoder(x)
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.lrelu(x)
        
        x = self.conv3(x)
        x = self.lrelu(x)
        x = self.dropout2(x)
        
        x = self.conv4(x)
        x = self.lrelu(x)
        x = self.conv5(x)
        return x
    
class MLP_Decoder(nn.Module):

    # the input shape is: 
    # ([1, 256, 64, 64])
    # 1024 / 16 = 64
    # hence each of the 16x16 patches is now represented as a 256 dimensional vector, 
    # there are 64x64 of these patches: 


    def __init__(self, p_dropout = 0.2):
        super().__init__()
        # fill in here: 
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
        # x has to be now of shape: (1,64,64,256 * 9)

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
        # x is now of shape: (1,64,64,256)
        x = x.reshape(batch_size, 64, 64, 16, 16)
        x = torch.swapaxes(x, 2, 3)
        # we want shape of x to be (batch_size, 1, 1024, 1024)
        x = x.reshape(batch_size, 1, 1024, 1024)
        #x = torch.swapaxes(x, 2, 3)
        return x
    

class Skip_MLP_Decoder(nn.Module):

    # the input shape is: 
    # ([1, 256, 64, 64])
    # 1024 / 16 = 64
    # hence each of the 16x16 patches is now represented as a 256 dimensional vector, 
    # there are 64x64 of these patches: 


    def __init__(self, p_dropout = 0.2):
        super().__init__()
        # fill in here: 
        self.dropout1 = nn.Dropout(p=p_dropout)
        self.dropout2 = nn.Dropout(p=p_dropout)
        self.dropout3 = nn.Dropout(p=p_dropout)
        self.dropout4 = nn.Dropout(p=p_dropout)
        self.dropout5 = nn.Dropout(p=p_dropout)

        self.fc1 = nn.Linear(256*10, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 256)

        self.gelu = nn.LeakyReLU()


    def forward(self, x):
        # input is of shape: dim_0 = batch_size,  ([1, 256, 64, 64])
        
        #print("skip decoder here, received x input", x, x[0].shape, x[1].shape)
        intermed = x[1]
        x = x[0]

        batch_size = x.shape[0]
        # torch.Size([1, 256, 64, 64]) torch.Size([1, 64, 64, 768, 3])

        x = torch.swapaxes(x, 1, 2)
        x = torch.swapaxes(x, 2, 3)
        # x is now of shape: (1,64,64,256)

        # reshape the intermediate output:
        intermed = intermed.reshape(batch_size, 64, 64, 768 * 3)
        # concat the intermediate output to the final output x:
        x = torch.cat((x, intermed), dim=3)
        # x has to be now of shape: (1,64,64,256 * 10)
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
        # x is now of shape: (1,64,64,256)
        x = x.reshape(batch_size, 64, 64, 16, 16)
        x = torch.swapaxes(x, 2, 3)
        # we want shape of x to be (batch_size, 1, 1024, 1024)
        x = x.reshape(batch_size, 1, 1024, 1024)
        #x = torch.swapaxes(x, 2, 3)
        return x

class MLP_Decoder_Spatially_Aware(nn.Module):
    # if option = 1, we use the 8 spatially aware patches, if option = 0 we use 4 spatially aware patches
    # the input shape is: 
    # ([1, 256, 64, 64])
    # 1024 / 16 = 64
    # hence each of the 16x16 patches is now represented as a 256 dimensional vector, 
    # there are 64x64 of these patches: 


    def __init__(self, p_dropout = 0.2, context_option = 1):
        super().__init__()
        # fill in here: 
        self.dropout1 = nn.Dropout(p=p_dropout)
        self.dropout2 = nn.Dropout(p=p_dropout)
        self.dropout3 = nn.Dropout(p=p_dropout)
        self.dropout4 = nn.Dropout(p=p_dropout)
        self.dropout5 = nn.Dropout(p=p_dropout)

        if (context_option == 1):
            self.fc1 = nn.Linear(256 * 9, 512)
        elif (context_option == 0):
            self.fc1 = nn.Linear(256 * 5, 512)
        else:
            raise ValueError("context_option should be either 0 or 1")

        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 256)

        self.gelu = nn.LeakyReLU()
        self.context_option = context_option

    def forward(self, x):
        # input is of shape: dim_0 = batch_size,  ([1, 256, 64, 64])
        batch_size = x.shape[0]
        
        x = torch.swapaxes(x, 1, 2)
        x = torch.swapaxes(x, 2, 3)
        # x is now of shape: (1,64,64,256)

        # ADD the spatially aware part:
        if self.context_option == 1:
            x_new = torch.zeros((batch_size, 64, 64, 256,9), device=x.device, requires_grad=False)
            x_new[:, :, :, :, 0] = x
            x_new[:, 1:, :, :, 1] = x[:, :-1, :, :]
            x_new[:, :-1, :, :, 2] = x[:, 1:, :, :]
            x_new[:, :, 1:, :, 3] = x[:, :, :-1, :]
            x_new[:, :, :-1, :, 4] = x[:, :, 1:, :]
            x_new[:, 1:, 1:, :, 5] = x[:, :-1, :-1, :]
            x_new[:, :-1, :-1, :, 6] = x[:, 1:, 1:, :]
            x_new[:, 1:, :-1, :, 7] = x[:, :-1, 1:, :]
            x_new[:, :-1, 1:, :, 8] = x[:, 1:, :-1, :]
            x = x_new
            x = x.reshape(batch_size, 64, 64, 256 * 9)
        elif self.context_option == 0:
            x_new = torch.zeros((batch_size, 64, 64, 256,5), device=x.device, requires_grad=False)
            x_new[:, :, :, :, 0] = x
            x_new[:, 1:, :, :, 1] = x[:, :-1, :, :]
            x_new[:, :-1, :, :, 2] = x[:, 1:, :, :]
            x_new[:, :, 1:, :, 3] = x[:, :, :-1, :]
            x_new[:, :, :-1, :, 4] = x[:, :, 1:, :]
            x = x_new
            x = x.reshape(batch_size, 64, 64, 256 * 5)
        else:
            raise ValueError("context_option should be either 0 or 1")

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
        # x is now of shape: (1,64,64,256)
        x = x.reshape(batch_size, 64, 64, 16, 16)
        x = torch.swapaxes(x, 2, 3)
        # we want shape of x to be (batch_size, 1, 1024, 1024)
        x = x.reshape(batch_size, 1, 1024, 1024)
        #x = torch.swapaxes(x, 2, 3)
        return x
    

class SAM_Encoder_Custom_Decoder(nn.Module):
    def __init__(self, sam_preprocess, sam_encoder, decoder):
        super().__init__()
        self.sam_preprocess = sam_preprocess
        self.sam_encoder = sam_encoder
    
        self.decoder = decoder

    def forward(self, x):
        x = self.sam_preprocess(x)
        x = self.sam_encoder(x)
        x = self.decoder(x)
        return x
    

class EnsembleMLP(nn.Module):

    def __init__(self, input_dim, p_dropout = 0.2):
        super().__init__()
        self.input_dim = input_dim


        self.fc1 = nn.Linear(input_dim, 1)
        #self.fc2 = nn.Linear(20, 20)
        #self.fc3 = nn.Linear(20, 1)

        #self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        # assume input is of shape: 
        # (batch_size,1024,1024,input_dim) and contains numbers between 0 and 1 (they have been feed trough a sigmoid)
        
        x = self.fc1(x)
        #x = self.lrelu(x)
        #x = self.dropout1(x)

        #x = self.fc2(x)
        #x = self.lrelu(x)
        #x = self.dropout2(x)

        #x = self.fc3(x)
        
        return x