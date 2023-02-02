from encoder import EncoderCNN
from decoder import DecoderLSTM
import setup

from train_and_test import cnn_lstm 
# from torch import nn
modelName = "cnn_lstm"
index_file = "../data/nasdaq100_padding.csv"


#                  file_data, decoder_hidden_size = 64, T = 10,
#                  input_dim= 1, channel_size = 64, feature_size = 81,
#                  learning_rate = 0.01, batch_size = 128, parallel = False, debug = False)
model = cnn_lstm(file_data=index_file, decoder_hidden_size = 100, T = 50,
                 input_dim = 1, channel_size = 64, feature_size = 81,
                 learning_rate = 0.01, batch_size = 128, parallel = False, debug = False)


model.train(n_epochs = 100)

y_train = model.predict(on_train=True)
y_pred = model.predict(on_train=False)

plt.figure()
plt.semilogy(range(len(model.iter_losses)), model.iter_losses)
plt.pause(0.1) 
plt.savefig("./results/%s/iter_losses_%s.png" %(modelName, modelName), bbox_inches="tight")

plt.figure()
plt.pause(0.001)
plt.semilogy(range(len(model.epoch_losses)), model.epoch_losses)
plt.pause(0.1) 
plt.savefig("./results/%s/epoch_losses_%s.png" %(modelName, modelName), bbox_inches="tight")