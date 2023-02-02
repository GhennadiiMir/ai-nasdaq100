import commons
import pandas as pd
from encoder import EncoderCNN
from decoder import DecoderLSTM
import setup
import torch
from torch import optim
from torch import nn
import numpy as np
import logging
import os
from torch.autograd import Variable
import matplotlib.pyplot as plt

modelName = "cnn_lstm"
# Train the model

class cnn_lstm:
    def __init__(self, file_data, decoder_hidden_size = 64, T = 10,
                 input_dim= 1, channel_size = 64, feature_size = 81,
                 learning_rate = 0.01, batch_size = 128, parallel = False, debug = False):
        self.T = T
        dat = pd.read_csv(file_data, nrows = 100 if debug else None)
        print("Shape of data: %s.\nMissing in data: %s" %(dat.shape, dat.isnull().sum().sum()))

        self.X = dat[[x for x in dat.columns if x != 'NDX']][:-1].to_numpy()
        # drop last row since using forward y
        y = dat.NDX.shift(-1).values
        self.y = y[:-1].reshape((-1, 1))
        
        print("X shape", self.X.shape)
        print("y shape", self.y.shape)
        
        self.batch_size = batch_size


        use_cuda = torch.cuda.is_available()
        if use_cuda: 
            #input_dim=1, channel_size=64, batch_size=10, T=100, feature_size=81)
            self.encoder = EncoderCNN(input_dim=input_dim, channel_size=channel_size,
                                     batch_size=batch_size, T=T, feature_size=feature_size).cuda()
            # feature_size, decoder_hidden_size, T=100, num_layers=2)
            self.decoder = DecoderLSTM(feature_size, decoder_hidden_size, T=T, num_layers=2).cuda()

        else:
            self.encoder = EncoderCNN(input_dim=input_dim, channel_size=channel_size,
                                        batch_size=batch_size, T=T, feature_size=feature_size).cpu()
            self.decoder = DecoderLSTM(feature_size, decoder_hidden_size, T=T, num_layers=2).cpu()


        if parallel:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)

        self.encoder_optimizer = optim.Adam(params = filter(lambda p: p.requires_grad, self.encoder.parameters()),
                                           lr = learning_rate)
        self.decoder_optimizer = optim.Adam(params = filter(lambda p: p.requires_grad, self.decoder.parameters()),
                                           lr = learning_rate)

        self.train_size = int(self.X.shape[0] * 0.7)
        self.train_mean = np.mean(self.y[:self.train_size])
        print("train mean: %s" %(self.train_mean))
        self.y = self.y - self.train_mean # Question: why Adam requires data to be normalized?
        print("Training size: %s" %(self.train_size))
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        logger.info("Training size: %s" %(self.train_size))

    def train(self, n_epochs = 10):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        iter_per_epoch = int(np.ceil(self.train_size * 1. / self.batch_size))
        print("Iterations per epoch: %s ~ %s" %(self.train_size * 1. / self.batch_size, iter_per_epoch))
        logger.info("Iterations per epoch: %s ~ %s" %(self.train_size * 1. / self.batch_size, iter_per_epoch))
        self.iter_losses = np.zeros(n_epochs * iter_per_epoch)
        self.epoch_losses = np.zeros(n_epochs)

        self.loss_func = nn.MSELoss()

        n_iter = 0

        learning_rate = 1.

        for i in range(n_epochs):
            print("\n-------------------------------------------")
            print("Epoch: ", i)
            logger.info("\n-------------------------------------------")
            logger.info("Epoch: %s" %(i))
            perm_idx = np.random.permutation(self.train_size - self.T)
            j = 0
            while j < self.train_size - self.T:
                batch_idx = perm_idx[j:(j + self.batch_size)]
                X = np.zeros((len(batch_idx), self.T, self.X.shape[1]))
                y_history = np.zeros((len(batch_idx), self.T))
                y_target = self.y[batch_idx + self.T]

                for k in range(len(batch_idx)):
                    X[k, :, :] = self.X[batch_idx[k] : (batch_idx[k] + self.T), :]
                    # y_history[k, :] (T-1,)
                    y_history[k, :] = self.y[batch_idx[k] : (batch_idx[k] + self.T)].flatten()

                # train
                loss = self.train_iteration(X, y_history, y_target)
#                 print("loss: ", loss.item())

                self.iter_losses[int(i * iter_per_epoch + j / self.batch_size)] = loss
                if (j / self.batch_size) % 50 == 0:
                    print("\tbatch: %s  loss: %s" %(j / self.batch_size, loss))
                    logger.info("\tbatch: %s  loss: %s" %(j / self.batch_size, loss))
                
                j += self.batch_size
                n_iter += 1

                # decrease learning rate
                if n_iter % 10000 == 0 and n_iter > 0:
                    for param_group in self.encoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9
                    logger.info("encoder learning rate: ", param_group["lr"])
                    for param_group in self.decoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9
                    logger.info("decoder learning rate: ", param_group["lr"])

                    
            self.epoch_losses[i] = np.mean(self.iter_losses[range(i * iter_per_epoch, (i + 1) * iter_per_epoch)])
            if i % 10 == 0:
                print("Epoch %s, loss: %s" %(i, self.epoch_losses[i]))
                logger.info("Epoch %s, loss: %s" %(i, self.epoch_losses[i]))
                
            if i % 10 == 0:
                print("\n Predict")
                y_train_pred = self.predict(on_train = True)
                y_test_pred = self.predict(on_train = False)
                y_pred = np.concatenate((y_train_pred, y_test_pred))
                y_pred = y_train_pred
                plt.figure()
                plt.plot(range(1, 1 + len(self.y)), self.y, label = "True")
                plt.plot(range(self.T , len(y_train_pred) + self.T), y_train_pred, label = 'Predicted - Train')
                plt.plot(range(self.T + len(y_train_pred) , len(self.y) + 1), y_test_pred, label = 'Predicted - Test')
                plt.legend(loc = 'upper left')
                plt.pause(0.1) 
                plt.savefig("./results/%s/predict_%s_epoch%s.png" %(modelName, modelName, i), bbox_inches="tight")

    def train_iteration(self, X, y_history, y_target):

        # zero gradient - original code placemenet
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        
        # define variables
        # if use_cuda:       
        #     Xt = Variable(torch.from_numpy(X).type(torch.FloatTensor).cuda())
        #     yht = Variable(torch.from_numpy(y_history).type(torch.FloatTensor).cuda())
        #     y_true = Variable(torch.from_numpy(y_target).type(torch.FloatTensor).cuda())            
        # else:
        Xt = Variable(torch.from_numpy(X).type(torch.FloatTensor).cpu())
        yht = Variable(torch.from_numpy(y_history).type(torch.FloatTensor).cpu())
        y_true = Variable(torch.from_numpy(y_target).type(torch.FloatTensor).cpu())
        

        # run models get prediction
        # Xt (N, C, H, W)
        Xt = Xt.view(Xt.size(0), 1, Xt.size(1), Xt.size(2))
        # yht (N, T, 1)
        yht = yht.unsqueeze(2)
        features = self.encoder(Xt)
        y_pred = self.decoder(features, yht)
        
        # loss 
        loss = self.loss_func(y_pred, y_true)
        loss.backward()
        
        # optimizer
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item()

    def predict(self, on_train = False):
        
        
        if on_train:
            y_pred = np.zeros(self.train_size - self.T + 1)
            print("PREDICT train")
        else:
            y_pred = np.zeros(self.X.shape[0] - self.train_size)
            print("PREDICT test")

        i = 0
        while i < len(y_pred):
            batch_idx = np.array(range(len(y_pred)))[i : (i + self.batch_size)]
            X = np.zeros((len(batch_idx), self.T, self.X.shape[1]))
            y_history = np.zeros((len(batch_idx), self.T))


            for j in range(len(batch_idx)):
                if on_train:
                    X[j, :, :] = self.X[range(batch_idx[j], batch_idx[j] + self.T), :]
                    y_history[j, :] = self.y[range(batch_idx[j],  batch_idx[j]+ self.T)].flatten()
                else:
                    X[j, :, :] = self.X[range(batch_idx[j] + self.train_size - self.T, batch_idx[j] + self.train_size), :]
                    y_history[j, :] = self.y[range(batch_idx[j] + self.train_size - self.T,  batch_idx[j]+ self.train_size)].flatten()
            
            # if use_cuda:
            #     Xt = Variable(torch.from_numpy(X).type(torch.FloatTensor).cuda())
            #     yht = Variable(torch.from_numpy(y_history).type(torch.FloatTensor).cuda())
            #     # Xt (N, C, H, W)
            #     Xt = Xt.view(Xt.size(0), 1, Xt.size(1), Xt.size(2))
            #     # yht (N, T, 1)
            #     yht = yht.unsqueeze(2)
            #     features = self.encoder(Xt)
            #     pred_cuda = self.decoder(features, yht)
            #     y_pred[i:(i+self.batch_size)] = pred_cuda.cpu().data.numpy()[:, 0]
            # else:
            Xt = Variable(torch.from_numpy(X).type(torch.FloatTensor).cpu())
            yht = Variable(torch.from_numpy(y_history).type(torch.FloatTensor).cpu())
            # Xt (N, C, H, W)
            Xt = Xt.view(Xt.size(0), 1, Xt.size(1), Xt.size(2))
            # yht (N, T, 1)
            yht = yht.unsqueeze(2)
            features = self.encoder(Xt)        
            y_pred[i:(i + self.batch_size)] = self.decoder(features, yht).data.numpy()[:, 0]           
            
            i += self.batch_size
        
        return y_pred