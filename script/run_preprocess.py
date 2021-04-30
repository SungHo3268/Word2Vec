import argparse
from distutils.util import strtobool as _bool
import os
import sys

sys.path.append(os.getcwd())
import json
import pickle
import gzip
from src.preprocess import make_dict, make_sentences

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--f_start', type=int, default=1, help="1~99")
parser.add_argument('--f_end', type=int, default=99, help='1~99')
parser.add_argument('--least_freq', type=int, default=5)

args = parser.parse_args()

with open(os.path.join('dataset/1-billion-word/preprocessed','preprocess_argparse.json'), 'w') as f:
    json.dump(args.__dict__, f)


# make dictionary
word_to_id, id_to_word, vocabulary = make_dict(args.f_start, args.f_end)
with open('dataset/1-billion-word/preprocessed/dictionary_all.pkl', 'wb') as fw:
    pickle.dump((word_to_id, id_to_word, vocabulary), fw)
print("Succeed to save (word_to_id, id_to_word, vocabulary) to 'dictionary_all.pkl'.")
print("vocabulary length : ", len(vocabulary))


with open('dataset/1-billion-word/preprocessed/dictionary_all.pkl', 'rb') as fr:
    word_to_id, id_to_word, vocabulary = pickle.load(fr)

# make sentence list
batch = args.f_end//5
for i in range(5):
    if i ==0:
        sentence_list = make_sentences(word_to_id, args.f_start, (i+1)*batch)
    else:
        sentence_list = make_sentences(word_to_id, i*batch, (i+1)*batch)

    with open('dataset/1-billion-word/preprocessed/sentence_list/sentence_list{}.pkl'.format(i), 'wb') as fw:
        pickle.dump(sentence_list, fw)
    print("Succeed to make sentence list{}".format(i))
    print("The length of the sentence list{} is ".format(i), len(sentence_list))

