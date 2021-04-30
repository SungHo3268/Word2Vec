import numpy as np
import os
import sys
import time
import pickle
sys.path.append(os.getcwd())
from src.utils import *

class CBOW:
    def __init__(self, V, D, activation, UnigramTable, negative, codeIndex, code_sign):
        self.D = D
        self.mode = activation
        self.UnigramTable = UnigramTable
        self.negative = negative
        self.codeIndex = codeIndex
        self.code_sign = code_sign

        if self.mode == 'softmax':
            self.W_in = 0.01*np.random.randn(V, D).astype('f')
            self.W_out = np.zeros((D, V)).astype('f')
        elif self.mode == 'NEG':
            self.W_in = 0.01*np.random.randn(V, D).astype('f')
            self.W_out = np.zeros((V, D)).astype('f')
        elif self.mode == 'HS':
            # #### load data
            # file = "log/CBOW_HS_sub1e-05_3epoch/weight/(first)CBOW_HS_sub1e-05_3epoch.pkl"
            # with open(file, 'rb') as fr:
            #     self.W_in, self.W_out = pickle.load(fr)

            self.W_in = 0.01*np.random.randn(V, D).astype('f')
            self.W_out = np.zeros((V-1, D)).astype('f')
        self.cache = None

    def forward(self, center, contexts):
        x = self.W_in[contexts]         # x = (contexts, D)
        x = np.sum(x, axis=0)           # x= (D, )
        x /= len(contexts)
        x = x.reshape(1, self.D)            # x = (1, D)

        # get negative samples
        if self.mode == 'NEG':
            neg_samples = []
            while 1:
                b = np.random.randint(low=0, high=len(self.UnigramTable), size=self.negative)
                if center in self.UnigramTable[b]:
                    continue
                else:
                    neg_samples = self.UnigramTable[b]
                    break
            label = np.append([center], neg_samples)

            # forward positive
            out = sigmoid(np.dot(x, self.W_out[label].T))       # out = (1, label)
            p_loss = -np.log(out[:, :1] + 1e-07)                    # p_loss = (1, )
            n_loss = -np.sum(np.log(1-out[:, 1:] + 1e-07))         # n_loss = (1, )

            self.cache = (x, contexts, label, out)
            return float(p_loss + n_loss), np.sum(out)/len(out)

        elif self.mode == 'HS':
            # forward
            code_index = self.codeIndex[center]
            sign = self.code_sign[center]
            score = np.dot(x, self.W_out[code_index].T)         # (1, path)
            score *= sign
            loss = -np.sum(np.log(sigmoid(score) + 1e-07))

            self.cache = (x, contexts, code_index, sign, score)
            return loss, np.sum(score)
        else:
            return 0

    def backward(self, lr):
        if self.mode == 'NEG':
            x, contexts, label, out = self.cache

            # backward
            dout = out.copy()                       # dout = (1, label)
            dout[:, :1] -= 1                            # y - t = p_out - 1 = (1, )
            dW_out = np.dot(x.T, dout).T            # dW_out = (label, D)
            dx = np.dot(dout, self.W_out[label])    # dx = (1, D)

            # update
            self.W_out[label] -= dW_out * lr
            self.W_in[contexts] -= dx.squeeze()/len(contexts) *lr
            return None

        elif self.mode == 'HS':
            x, contexts, code_index, sign, score = self.cache
            # backward
            dout = sigmoid(score)
            dout -= 1
            dout *= sign
            dx = np.dot(dout, self.W_out[code_index])       # dx = (1, D)
            dW_out = np.dot(x.T, dout).T                    # dW_out = (code_index, D)

            # update
            self.W_out[code_index] -= dW_out * lr
            self.W_in[contexts] -= dx.squeeze()/len(contexts) * lr
            return None
        else:
            return None


class SkipGram:
    def __init__(self, V, D, activation, UnigramTable, negative, codeIndex, code_sign):
        self.D = D
        self.mode = activation
        self.UnigramTable = UnigramTable
        self.negative = negative
        self.codeIndex = codeIndex
        self.code_sign = code_sign

        if self.mode == 'softmax':
            self.W_in = 0.01*np.random.randn(V, D).astype('f')
            self.W_out = np.zeros((D, V)).astype('f')
        elif self.mode == 'NEG':
            self.W_in = 0.01*np.random.randn(V, D).astype('f')
            self.W_out = np.zeros((V, D)).astype('f')
        elif self.mode == 'HS':
            self.W_in = 0.01*np.random.randn(V, D).astype('f')
            self.W_out = np.zeros((V-1, D)).astype('f')
        self.pos_len = 0
        self.cache = None

    def forward(self, center, contexts):
        if self.mode == 'NEG':
            neg_samples = []
            cache = []
            loss = 0
            total_out = []
            for context in contexts:
                #get negative samples
                while 1:
                    b = np.random.randint(low=0, high=len(self.UnigramTable), size=self.negative)
                    neg_sample = self.UnigramTable[b]
                    if center in neg_sample:
                        continue
                    else:
                        neg_samples.append(neg_sample)
                        break
                target = np.append(center, neg_sample)      # center word is a positive sample.

                x = self.W_in[context]
                x = x.reshape(1, self.D)
                # forward
                out = sigmoid(np.dot(x, self.W_out[target].T))         # out = (1, label)   at label=(contexts+neg_samples)
                total_out.append(np.sum(out)/len(out))
                p_loss = -np.log(out[:, :1] + 1e-07)
                n_loss = -np.sum(np.log(1-out[:, 1:] + 1e-07))             # loss = (1, )
                loss += (p_loss + n_loss)
                cache.append((x, context, target, out))

            self.cache = cache
            return float(loss)/len(contexts), np.sum(total_out)/len(total_out)

        elif self.mode == 'HS':
            code_index = self.codeIndex[center]                 # code_index = (code_length, )
            sign = self.code_sign[center]                       # sign = (code_length, )

            xs = self.W_in[contexts]                            # xs = (contexts, D)
            score = np.dot(xs, self.W_out[code_index].T)        # score = (contexts, code_length)
            score *= sign
            loss = -np.sum(np.log(sigmoid(score) + 1e-07))      # loss = (1, )
            loss /= len(contexts)

            self.cache = (xs, contexts, code_index, sign, score)
            return loss, np.sum(score)
        else:
            return 0

    def backward(self, lr):
        if self.mode == 'NEG':
            for x, context, target, out in self.cache:
                # backward
                dout = out.copy()                       # dout = (1, target)
                dout[:, :1] -= 1                        # subtract positive label=1
                dW_out = np.dot(x.T, dout).T            # dW_out = (target, D)
                dx = np.dot(dout, self.W_out[target])    # dx = (1, D)

                # update
                self.W_out[target] -= dW_out * lr
                self.W_in[context] -= dx.squeeze() *lr

        elif self.mode == 'HS':
            xs, contexts, code_index, sign, score = self.cache
            # omit dividing loss by len(contexts) because I just interest in gradient not value.
            dout = sigmoid(score)            # dout = (contexts, code_length)
            dout -= 1
            dout *= sign

            dx = np.dot(dout, self.W_out[code_index])       # dx = (contexts, D)
            dW_out = np.dot(xs.T, dout).T                   # dW_out = (code_length, D)

            self.W_out[code_index] -= dW_out * lr
            self.W_in[contexts] -= dx * lr
            return None
        else:
            return None
