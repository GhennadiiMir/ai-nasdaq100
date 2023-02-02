import commons
import setup
from torch import nn
import logging
import os
import torch

class DecoderLSTM(nn.Module):
    def __init__(self, feature_size, decoder_hidden_size, T=100, num_layers=2):
        super(DecoderLSTM, self).__init__()

        self.T = T
        self.feature_size = feature_size
        self.decoder_hidden_size = decoder_hidden_size
#         print("decoder: decoder_hidden_size: ", decoder_hidden_size)
        
        # lstm - in: (N, T, W) out: (N, T, H)
        self.lstm_layer = nn.LSTM(feature_size, decoder_hidden_size, 
                                  num_layers=num_layers,
                                  dropout=0.2, 
                                  batch_first=True)
        # dense layer - in: (N, T*H), out: (N, T*H)
        self.dense_layer = nn.Sequential(
            nn.Linear(T*(decoder_hidden_size+1), T*(decoder_hidden_size+1)),
            nn.ReLU(),
            nn.Dropout(0.2))
        # final layer - in: (N, T*H) out:(N, 1)
        self.final_layer = nn.Linear(T*(decoder_hidden_size+1), 1)

        # log info
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        logger.info("decoder - feature_size: %s hidden_size: %s T: %s" \
                    %(feature_size, decoder_hidden_size, T))
        
    def forward(self, features, y_history):
        
        # x (N, T, W) y_history (N, T, 1)
#         print("features: ", features.size())
#         print("y_history: ", y_history.size())
  
        # lstm layer in: (N, T, W) out: (N, T, H)
        out, lstm_out = self.lstm_layer(features)
#         print("lstm layer: ", out.size())
    
        # clipping to eliminate nan's from lstm
        out.register_hook(lambda x: x.clamp(min=-100, max=100))
        
        # combine with y_history
        out = torch.cat((out, y_history), dim=2)
#         print("out cat: ", out.size())
        
        # flatten in: (N, T, H) out: (N, T*(H=1))
        out = out.contiguous().view(-1, out.size(1)*out.size(2))
#         print("out flatten: ", out.size())
               
        # final layer in: (N, T*(H+1)), out: (N, 1)
        out = self.final_layer(out)
#         print("final layer: ", out.size())
        
        return out

    def init_hidden(self, x):
        return Variable(x.data.new(1, x.size(0), self.decoder_hidden_size).zero_())