from torch import nn
import pandas as pd

class EncoderCNN(nn.Module):
    def __init__(self, input_dim=1, channel_size=64, batch_size=10, T=100, feature_size=81):
        super(EncoderCNN, self).__init__()
        self.input_dim = input_dim    # num channels
        self.batch_size = batch_size
        self.channel_size = channel_size
        self.T = T
        self.feature_size = feature_size
        # (N, C, H, W) = (num_batch, features, history, stocks)
        # Conv2d - out:(N, 64, 100, 81), kernel(3, 5) stride:1
        
        # added a linear layer to shrink the num stocks lower due to memory 
        self.small_feature_size = 10
        self.first_linear = nn.Linear(feature_size, self.small_feature_size)
        
        self.first_cnn_layer = nn.Sequential(
            nn.Conv2d(input_dim, channel_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2))
        
        # Conv2d - out:(N, 64, 100, 81), kernel(3,5) stride:1
        self.second_cnn_layer = nn.Sequential(
            nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2))

        # dense layer - in: (N, 100*64*81), out: (N, 100*81)
        self.first_dense_layer = nn.Sequential(
            nn.Linear(T*self.small_feature_size*channel_size, T*self.feature_size),
            nn.ReLU(),
            nn.Dropout(0.2))

    
    def forward(self, xt):
        # conv2d input (N, 1, H, W) expects (N, C, H, W) 
#         print("x: ", xt.size())
        N = xt.size(0)
        
        # lin: in (N, 1, H, W) out: (N, 1, H, 10)
        out = self.first_linear(xt)
#         print("cnn: linear: ", out.size())
        
        # cnn: in (N, 1, H, 10) out: (N, C, H, 10)
        out = self.first_cnn_layer(out)
#         print("cnn: first_layer output: ", out.size())

        # cnn: in (N, C, H, 10) out: (N, C, H, 10)
        out = self.second_cnn_layer(out)
#         print("cnn: second_layer output: ", out.size())

        # reshape for linear layer
        out = out.view(N, self.T*self.small_feature_size*self.channel_size)
#         print("flatten: ", out.size())

        # first dense layer in: (N, C*H*W) out: (N, H*W)
        out = self.first_dense_layer(out)
#         print("first dense layer output: ", out.size())
        
        # reshape output for (N, T, W)
        out = out.reshape(out.size(0), self.T, self.feature_size)
#         print("reshape out: ", out.size())

        return out