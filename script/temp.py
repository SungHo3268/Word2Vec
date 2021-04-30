import argparse
from distutils.util import strtobool as _bool
from torch.utils.tensorboard import SummaryWriter
import pickle
import gzip
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

# load dictionary
file = 'dataset/1-billion-word/preprocessed/dictionary_all.pkl'
word_to_id, id_to_word, vocabulary = load_dict(file)
V = len(vocabulary)

# make Unigram Table
UnigramTable, pos_index = unigramTable(vocabulary, word_to_id)
UnigramTable = np.array(UnigramTable)

# make random value to get temporary context.
context_list = []
for _ in range(1000):
    a = np.random.randint(low=0, high=len(vocabulary), size=1)
    if a == 0:
        continue
    else:
        context_list.append(a)
context_list = np.array(context_list).reshape(-1, 5)


############################### randint_1 (compare word) ###############################
neg_samples = []
negative = 5
t_list = []
for contexts in tqdm(context_list):
    start_t = time.time()
    for context in contexts:
        neg_count = 0
        while 1:
            b = np.random.randint(low=0, high=len(UnigramTable), size=1)
            if UnigramTable[int(b)] == id_to_word[int(context)]:
                continue
            else:
                neg_samples.append(word_to_id[UnigramTable[int(b)]])
                neg_count+=1
                if neg_count == negative:
                    break
    elapsed_t = time.time() - start_t
    t_list.append(elapsed_t)
avg_t = sum(t_list)/len(t_list)
print("randint_1 (compare word): ", avg_t)


############################### randint_1 (compare range) ###############################
neg_samples = []
negative = 5
t_list = []
for contexts in tqdm(context_list):
    start_t = time.time()
    for context in contexts:
        neg_count = 0
        while 1:
            b = np.random.randint(low=0, high=len(UnigramTable), size=1)
            if pos_index[context][0] <= b < pos_index[context][1]:
                continue
            else:
                neg_samples.append(word_to_id[UnigramTable[int(b)]])
                neg_count+=1
                if neg_count == negative:
                    break
    elapsed_t = time.time() - start_t
    t_list.append(elapsed_t)
avg_t = sum(t_list)/len(t_list)
print("randint_1 (compare range): ", avg_t)


############################### randint_5 ###############################
neg_samples = []
negative = 5
t_list = []
for contexts in tqdm(context_list):
    start_t = time.time()
    for context in contexts:
        neg_count = 0
        while 1:
            b = np.random.randint(low=0, high=len(UnigramTable), size=5)
            if id_to_word[int(context)] in UnigramTable[b]:
                continue
            else:
                neg_samples.append(word_to_id[UnigramTable[int(b)]])

    elapsed_t = time.time() - start_t
    t_list.append(elapsed_t)
avg_t = sum(t_list)/len(t_list)
print("randint_5 (without compare context): ", avg_t)



