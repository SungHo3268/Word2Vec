import pickle
import gzip
import numpy as np
from tqdm.auto import tqdm


def load_dict(file):
    with open(file, 'rb') as fr:
        word_to_id, id_to_word, vocabulary = pickle.load(fr)
    print("vocabulary size: ", len(vocabulary))
    return word_to_id, id_to_word, vocabulary


def load_data(file):
    with gzip.open(file, 'rb') as fr:
        sentence_list = pickle.load(fr)
    print("load training dataset.")
    return sentence_list


def count_total_word(sentence_list):
    total_word = 0
    for sentence in tqdm(sentence_list, desc='count words', bar_format="{l_bar}{bar:10}{r_bar}"):
        for _ in sentence:
            total_word += 1
    print("total training word: ", total_word)
    return total_word


def subsampling_prob(vocabulary, word_to_id, total_word, sub_t):
    key = []
    idx = []
    prob = []
    sub_p = []
    for (word, f) in vocabulary.items():
        key.append(word)
        idx.append(word_to_id[word])
        prob.append(f/total_word)
    for p in prob:
        sub_p.append((1+np.sqrt(p/sub_t)) * sub_t/p)
    id_to_sub_p = dict(np.stack((idx, sub_p), axis=1))
    return id_to_sub_p


def unigramTable(vocabulary, word_to_id):
    UnigramTable = []   # sampling pool = Unigram table.       len(UnigramTable) = total_word
    current = 0         # the current index of sub_list
    pos_index = []      # the start index of words with word(char) in UnigramTable.
    for word in tqdm(vocabulary.keys(), desc='Making UnigramTable', bar_format="{l_bar}{bar:10}{r_bar}"):
        if word == '\s':
            continue
        freq = int(pow(vocabulary[word], 3/4))
        pos_index.append((current, current+freq))   # (start_idx, next_start_idx)
        current += freq         # It's for the next word's index
        temp = [word_to_id[word]] * freq
        UnigramTable.extend(temp)
    print("UnigramTable length applied 3/4 pow: ", len(UnigramTable))
    return UnigramTable, pos_index


def subsampling(sentence_, id_to_sub_p):
    sentence = []
    for word in sentence_:
        # double check, if center word is '\s', skip
        if word == 0: continue
        if id_to_sub_p[word] > np.random.random():
            sentence.append(word)
    return sentence


def make_contexts(window_size, sentence, c):        # c means 'current sentence position'
    # make random window_size
    b = int(np.random.randint(1, window_size+1))

    # make contexts by shrunk args.window_size.
    contexts = []
    for j in range(-b, b+1):
        # only take account into within boundaries.
        cur = c+j
        if cur < 0:
            continue
        elif cur == c:     # if j==0
            continue
        elif cur >= len(sentence):
            break
        else:       # cur(=current_index) is among sentence.
            if sentence[cur] == 0:
                continue
            else:
                contexts.append(sentence[cur])         # complete contexts
    return contexts


def code_to_id(codes, root, vocabulary):
    node = root
    idx = []
    code_sign = []
    for word in tqdm(vocabulary, desc='get codes to index', bar_format="{l_bar}{bar:10}{r_bar}"):
        temp0 = []
        temp1 = []
        code = codes[word]
        for c in code:
            if c == '0':
                temp0.append(node.index)
                temp1.append(-1)
                node = node.left
            elif c == '1':
                temp0.append(node.index)
                temp1.append(1)
                node = node.right
            if node.index is None:
                node = root
                break
        idx.append(temp0)
        code_sign.append(temp1)
    return idx, code_sign

####################################################################
########################  Analogy functions ########################
####################################################################
def checkValid(question, vocabulary):
    valid_que = []
    for que in question:
        if que[0] in vocabulary and que[1] in vocabulary and que[2] in vocabulary and que[3] in vocabulary:
            valid_que.append(que)
    return valid_que


def convert2vec(valid, word_vectors, word_to_id):
    a = []
    b = []
    c = []
    d = []
    for s in valid:
        a_temp = word_vectors[word_to_id[s[0]]]
        b_temp = word_vectors[word_to_id[s[1]]]
        c_temp = word_vectors[word_to_id[s[2]]]
        d_temp = word_vectors[word_to_id[s[3]]]
        
        a_norm = np.linalg.norm(a_temp)
        b_norm = np.linalg.norm(b_temp)
        c_norm = np.linalg.norm(c_temp)
        d_norm = np.linalg.norm(d_temp)
        
        a.append(a_temp/a_norm)
        b.append(b_temp/b_norm)
        c.append(c_temp/c_norm)
        d.append(d_temp/d_norm)
    return np.array(a), np.array(b), np.array(c), np.array(d)


def cos_similarity(predict, word_vectors):
    norm_predict = np.linalg.norm(predict, axis=1)
    norm_words = np.linalg.norm(word_vectors, axis=1)

    similarity = np.dot(predict, word_vectors.T)      # similarity = (N, V)
    similarity *= 1/norm_words
    similarity = similarity.T
    similarity *= 1/norm_predict
    similarity = similarity.T

    return similarity


def count_in_top4(similarity, id_to_word, valid):
    count = 0
    max_top4 = []
    sim_top4 = []
    for i in tqdm(range(len(similarity)), bar_format="{l_bar}{bar:10}{r_bar}"):
        max_arg = np.argsort(similarity[i])[::-1]
        temp = list(max_arg[:4])
        max_top4.append(temp)
        sim_top4.append(list(similarity[i][temp]))

        for j in range(4):
            pred = id_to_word[temp[j]]
            if pred in valid[i]:
                if pred == valid[i][3]:
                    count += 1
            else: break
    return max_top4, sim_top4, count

