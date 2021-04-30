import argparse
from distutils.util import strtobool as _bool
from torch.utils.tensorboard import SummaryWriter
import pickle
from tqdm.auto import tqdm
import time
import json
import os
import sys
sys.path.append(os.getcwd())
from src.Huffman import *
from src.utils import *
from src.models import *
from src.functions import *


################################ Arg parser ################################
parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--model', type=str, default='CBOW', help="SkipGram | CBOW")
parser.add_argument('--activation', type=str, default='HS', help="softmax | NEG | HS")
parser.add_argument('--least_freq', type=int, default=5)
parser.add_argument('--window_size', type=int, default=5)
parser.add_argument('--negative', type=int, default=5)
parser.add_argument('--max_epoch', type=int, default=3)
parser.add_argument('--lr', type=float, default=0.025)
parser.add_argument('--sub_t', type=float, default=1e-05)
parser.add_argument('--hidden_dim', type=int, default=300)
parser.add_argument('--eval_interval', type=int, default=10000)

args = parser.parse_args()
log_dir = 'log/{}_{}{}_sub{}_window{}_dim{}_{}epoch/'.format(args.model, args.activation, args.negative, args.sub_t, args.window_size, args.hidden_dim, args.max_epoch)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
with open(log_dir + 'argparse.json', 'w') as f:
    json.dump(args.__dict__, f)


############################## tensor board ##################################
tb_dir = os.path.join(log_dir, 'tb')
ckpt_dir = os.path.join(log_dir, 'ckpt')

if not os.path.exists(log_dir):
    os.mkdir(log_dir)
if not os.path.exists(tb_dir):
    os.mkdir(tb_dir)
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)

tb_writer = SummaryWriter(tb_dir)


############################## Init net #################################
# load dictionary
file = 'dataset/1-billion-word/preprocessed/dictionary_all.pkl'
word_to_id, id_to_word, vocabulary = load_dict(file)
V = len(vocabulary)


# load data     - count the number of sentences
sentence_num = 0
for i in range(5):
    with open('dataset/1-billion-word/preprocessed/sentence_list/sentence_list{}.pkl'.format(i), 'rb') as fr:
        sentence_list = pickle.load(fr)
        sentence_num += len(sentence_list)


# count total training words.       - total word = 765,887,516 words.
total_word = count_total_word(sentence_list)


# probability of subsampling
id_to_sub_p = {}
if args.sub_t > 0:
    id_to_sub_p = subsampling_prob(vocabulary, word_to_id, total_word, args.sub_t)


# UniGramSampling Table
UnigramTable=[]
pos_index = []
if args.activation == 'NEG':
    UnigramTable, pos_index = unigramTable(vocabulary, word_to_id)
    UnigramTable = np.array(UnigramTable)


# Huffman Tree
codeIndex = []
code_sign = []
if args.activation == 'HS':
    HC = HuffmanCoding()
    codes, root = HC.build(vocabulary)
    codeIndex, code_sign = code_to_id(codes, root, vocabulary)    # id is string because of sign.


############################### start train ##################################
# select the model
model = None
if args.model == 'CBOW':
    model = CBOW(V, args.hidden_dim, args.activation, UnigramTable, args.negative, codeIndex, code_sign)
elif args.model == 'SkipGram':
    model = SkipGram(V, args.hidden_dim, args.activation, UnigramTable, args.negative, codeIndex, code_sign)

# training start
real_train_word = 0
current = 0
total_score = 0
total_loss = 0
loss_count = 0

time_list = []
start_t = time.time()
for epoch in range(args.max_epoch):
    print("epoch: %d/%d" %(epoch+1, args.max_epoch))
    # per dataset segment
    for j in range(5):
        with open('dataset/1-billion-word/preprocessed/sentence_list/sentence_list{}.pkl'.format(j), 'rb') as fr:
            sentence_list = pickle.load(fr)
        # per sentence
        for i in tqdm(range(len(sentence_list)), desc='dataset {}/5'.format(j+1), bar_format="{l_bar}{bar:10}{r_bar}"):
            current += 1
            # Do subsampling.        return a sentence.
            if args.sub_t > 0:
                sentence = subsampling(sentence_list[i], id_to_sub_p)
                if not sentence:         # if all words discarded.
                    continue
            else: sentence = sentence_list[i]

            # per center word
            for c, center in enumerate(sentence):
                # t0 = time.time()
                real_train_word += 1

                # apply decreasing learning rate
                alpha = 1 - current/ (sentence_num*args.max_epoch)
                if alpha <= 0.0001:
                    alpha = 0.0001
                lr = args.lr * alpha

                # make contexts by shrunk args.window_size.
                contexts = make_contexts(args.window_size, sentence, c)
                if not contexts:
                    continue

                # start real training.
                loss, score = model.forward(center, contexts)
                model.backward(lr)      # it includes update the gradient to parameters.

                total_loss += loss
                total_score += score
                loss_count += 1

                if (args.eval_interval is not None) and (real_train_word%args.eval_interval ==1):
                    elapsed_t = time.time() - start_t
                    avg_loss = total_loss/loss_count
                    avg_score = total_score/loss_count

                    total_loss = 0
                    total_score = 0
                    loss_count = 0

                    tb_writer.add_scalar('score/real_train_word(*{})'.format(args.eval_interval), avg_score, real_train_word)
                    tb_writer.add_scalar('loss/real_train_word(*{})'.format(args.eval_interval), avg_loss, real_train_word)
                    tb_writer.add_scalar('lr/real_train_word(*{})'.format(args.eval_interval), lr, real_train_word)
                tb_writer.flush()
        # save temp weight per dataset segment
        weight_dir = os.path.join(log_dir, 'weight')
        if not os.path.exists(weight_dir):
            os.mkdir(weight_dir)
        with open(os.path.join(weight_dir, 'ckpt_weight.pkl'
                .format(args.model, args.activation, args.negative, args.sub_t, epoch+1)), 'wb') as fw:
            pickle.dump((model.W_in, model.W_out), fw)
        
        et = (time.time() - start_t)/3600
        time_list.append(et)
        print("epoch: {}/{}, dataset {}, elapsed_time: {}[h]".format(epoch+1, args.max_epoch, str(j+1)+'/5', et))
        start_t = time.time()
        
    print("real train words per epoch: ", real_train_word)

    # save the weights
    weight_dir = os.path.join(log_dir, 'weight')
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)
    with open(os.path.join(weight_dir, '{}_{}{}_sub{}_window{}_dim{}_{}epoch.pkl'
            .format(args.model, args.activation, args.negative, args.sub_t, args.window_size, args.hidden_dim, epoch+1)), 'wb') as fw:
        pickle.dump((model.W_in, model.W_out), fw)
