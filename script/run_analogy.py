import argparse
from distutils.util import strtobool as _bool
import pickle
import numpy as np
from tqdm.auto import tqdm
import os
import sys
sys.path.append(os.getcwd())
from src.functions import *


################################ Arg parser ################################
parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--logdir', type=str, default='SkipGram_NEG5_sub1e-05_3epoch/weight')
parser.add_argument('--weight', type=str, default='ckpt_weight.pkl')
parser.add_argument('--inORout', type=str, default='in')
parser.add_argument('--data_seg', type=int, default=10)

args = parser.parse_args()


############################### Init Net ##################################
# load analogy data
with open("dataset/1-billion-word/analogy/questions-words.txt", 'r') as fr:
    loaded = fr.readlines()
count = 0
semantic = []
syntactic = []
for line in loaded:
    if line[0] == ':':
        count += 1
        continue
    elif line == '\n':
        continue
    if count < 6:
        semantic.append(line.split())
    else:
        syntactic.append(line.split())

# load word vectors
file = os.path.join('log', args.logdir, args.weight)
with open(file, 'rb') as fr:
    vectors = pickle.load(fr)
word_vectors = None
if args.inORout == 'in':
    word_vectors = np.array(vectors[0])
elif args.inORout == 'out':
    word_vectors = np.array(vectors[1])

# load dictionary
with open("dataset/1-billion-word/preprocessed/dictionary_all.pkl", 'rb') as fr:
    word_to_id, id_to_word, vocabulary = pickle.load(fr)

# Check whether my word vectors contain all words in questions
valid_sem = checkValid(semantic, vocabulary)
valid_syn = checkValid(syntactic, vocabulary)
print("valid semantic: %d/%d" %(len(valid_sem), len(semantic)))
print("valid syntactic: %d/%d" %(len(valid_syn), len(syntactic)))


############################### Start evaluate ##################################
batch1 = len(valid_syn)//args.data_seg + 1
batch2 = len(valid_sem)//args.data_seg + 1
syn_counts = 0
sem_counts = 0
for i in range(args.data_seg):
    print('data: {}/{}'.format(i+1, args.data_seg))
    batch_syn = valid_syn[i*batch1: (i+1)*batch1]
    batch_sem = valid_sem[i*batch2: (i+1)*batch2]

    # syntactic
    a1, b1, c1, d1 = convert2vec(batch_syn, word_vectors, word_to_id)
    predict_syn = b1 - a1 + c1
    similarity_syn = cos_similarity(predict_syn, word_vectors)
    syn_max_top4, syn_sim_top4, syn_count = count_in_top4(similarity_syn, id_to_word, batch_syn)
    syn_counts += syn_count

    #semantic
    a2, b2, c2, d2 = convert2vec(batch_sem, word_vectors, word_to_id)
    predict_sem = b2 - a2 + c2
    similarity_sem = cos_similarity(predict_sem, word_vectors)
    sem_max_top4, sem_sim_top4, sem_count = count_in_top4(similarity_sem, id_to_word, batch_sem)
    sem_counts += sem_count

syn_acc = syn_counts/len(valid_syn) * 100
sem_acc = sem_counts/len(valid_sem) * 100
print("syntactic accuracy: ", syn_acc)
print("semantic accuracy: ", sem_acc)
print("total accuracy: ", (syn_acc*len(valid_syn) + sem_acc*len(valid_sem))/
                                                             (len(valid_syn)+len(valid_sem)))
