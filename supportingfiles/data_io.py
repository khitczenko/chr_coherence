from __future__ import print_function

import gensim
import numpy as np
import pickle
from os import listdir
from os.path import isfile, join
from allennlp.commands.elmo import ElmoEmbedder
from nltk import pos_tag
import pandas as pd
import sys

def getWordmap(textfile, embmodel):
    '''
    Given:
    :textfile: a file with pre-trained word embeddings
    :embmodel: --lsa, --glove, --word2vec, --elmo
    Return:
    :words: a dictionary, words['str'] is the index of the word vector of 'str' in We
    :We: a vector containing the word vectors (words tells us which index to access)
    '''
    words={}
    We = []
    # Different textfiles have different formats
    if embmodel in ['--lsa', '--glove']:
        f = open(textfile,'r')
        lines = f.readlines()
        curridx = 0
        for i in lines:
            i=i.split()
            if embmodel == '--glove' and len(i) != 301:
                continue
            try:
                float(i[1])
            except ValueError:
                print(i[1])
                continue
            j = 1
            v = []
            while j < len(i):
                v.append(float(i[j]))
                j += 1
            if i[0] == '"':
                if i[0] in words:
                    continue
                words[i[0]] = curridx
            else:
                if i[0].replace('"', '') in words:
                    continue
                words[i[0].replace('"', '')] = curridx
            We.append(np.array(v))
            curridx += 1
        return(words, np.array(We))

    elif embmodel in ['--word2vec', '--elmo']:
        model = gensim.models.KeyedVectors.load_word2vec_format(textfile, binary=True)
        vocab = model.index2word
        for ii in range(0, len(vocab)):
            words[vocab[ii]] = ii
            We.append(model[vocab[ii]])
        return(words, np.array(We))
    else:
        print("Unknown model")

def prepare_data(list_of_seqs):
    lengths = [len(s) for s in list_of_seqs] # list of how long each sentence is
    n_samples = len(list_of_seqs) # number of sentences
    maxlen = np.max(lengths) # maximum length of a sentence
    # create matrices for x,m: 1 row = 1 sentence; 1 column = 1 word in sentence
    x = np.zeros((n_samples, maxlen)).astype('int32')
    x_mask = np.zeros((n_samples, maxlen)).astype('float32')
    # Fill in matrices with sentence info
    for idx, s in enumerate(list_of_seqs):
        x[idx, :lengths[idx]] = s
        x_mask[idx, :lengths[idx]] = 1.
    x_mask = np.asarray(x_mask, dtype='float32')
    return x, x_mask

def lookupIDX(words,w):
    '''
    Given a word 'str', output its index in the words dictionary (i.e. position in We)
    : words is dict: words['str'] = idx of 'str' in We
    : w is a particular word in a sentence 
    '''
    w = w.lower()
    if len(w) > 1 and w[0] == '#':
        w = w.replace("#","")
    if w in words:
        return words[w]
    # w is not in words, either output idx of 'UUUNKKK' if that existss
    elif 'UUUNKKK' in words:
        return words['UUUNKKK']
    # or output the last element in words
    else:
        return len(words) - 1

def getSeq(p1,words):
    """
    Given a sentence and the word indices, output the index for each word in words
    : p1 is a sentence (one element of sentences)
    : words is dict: words['str'] = idx of 'str' in We
    : X1 is a list containing the index of each word in p1 in words
    """
    p1 = p1.split()
    X1 = []
    for i in p1:
        X1.append(lookupIDX(words,i))
    return X1

def sentences2idx(sentences, words):
    """
    Given a list of sentences, output array of word indices that can be fed into the algorithms.
    :param sentences: a list of sentences
    :param words: a dictionary, words['str'] is the indices of the word 'str'
    :return: x1, m1. x1[i, :] is the word indices in sentence i, m1[i,:] is the mask for sentence i (0 means no word at the location)
    """
    seq1 = []
    for i in sentences:
        seq1.append(getSeq(i,words))
    # seq1 is a list of lists containing the index of each word in the sentences (one lst/sentence)
    x1,m1 = prepare_data(seq1)
    return x1, m1

def getWordWeight(weightfile, denom, a=1e-3):
    '''
    This reads a weightfile to get a weight for each word contained within it
    Given:
    :weightfile: file with word frequencies in large corpus to determine weights (e.g. Wikipedia)
    :denom: What do we want to use as the denominator (e.g. do we want to use the max freq or the total freq?)
    :a: weighting parameter
    Return:
    :word2weight: word2weight['str'] is the weight for the word 'str'
    (Note: 'str' possibilities are decided by what is in the weight file which could differ from the pretrained word embeddings file)
    '''
    if denom not in ['total', 'max', 'freq']:
        print("Unrecognized denominator type")
        denom = 'total'
    if a <=0: # when the parameter makes no sense, use unweighted
        a = 1.0
    word2weight = {}
    with open(weightfile) as f:
        lines = f.readlines()
    N = 0
    for i in lines:
        i=i.strip()
        if(len(i) > 0):
            i=i.split()
            if(len(i) == 2):
                word2weight[i[0]] = float(i[1])
                if denom == 'total':
                    N += float(i[1])
                elif denom == 'max':
                    N = max(N, float(i[1]))
            else:
                print(i)
    if denom != 'freq':
        for key, value in word2weight.items():
            word2weight[key] = a / (a + value/N)
    return word2weight

def getWeight(words, word2weight):
    '''
    This function gets a weight for each word in words (i.e. that we have a pretrained embedding for). If a word is in words but
    not in word2weight, we just give it a weight of 1.0.
    Given:
    :words: words: a dictionary, words['str'] is the indices of the word 'str'
    :word2weight: word2weight['str'] is the weight for the word 'str'
    Note: the keys come from different files (pretrained embeddings vs Wikipedia, so this function merges them)
    Return:
    :weight4ind: weight4ind[i] is the weight for the i-th word (in words)
    '''
    weight4ind = {}
    for word, ind in words.items():
        if word in word2weight:
            weight4ind[ind] = word2weight[word]
        else:
            weight4ind[ind] = 1.0
    return weight4ind

def loadWordWeights(weightfile, words):
    '''
    This loads word weight information to be used for the SIF/TFIDF weighting schemes
    Given:
    :weightfile: file with word frequencies in large corpus to determine weights (e.g. Wikipedia)
    :words: a dictionary, words['str'] is the indices of the word 'str'
    Return:
    :word2weight: word2weight['str'] is the weight for the word 'str'
    :weight4ind[i] is the weight for the i-th word
    '''
    weightpara = 1e-3 # the parameter in the SIF/TFIDF weighting scheme, usually in the range [3e-5, 3e-3]
    # rmpc = 1 # number of principal components to remove in SIF weighting scheme
    # params = params.params()
    # params.rmpc = rmpc
    word2weight = getWordWeight(weightfile, 'total', weightpara)
    weight4ind = getWeight(words, word2weight) 
    return word2weight, weight4ind

def seq2weight(seq, mask, weight4ind):
    '''
    Get word weights from
    :param x: array of word indices in one participant's text passage
    :param m: array (same size as x) that is 1 when a word is there, 0 if no word there
    :weight4ind: weight for word with index i (where index comes from words)
    '''
    weight = np.zeros(seq.shape).astype('float32')
    for i in range(seq.shape[0]):
        for j in range(seq.shape[1]):
            # Anything with mask 0 has weight 0, get other weights from seq
            if mask[i,j] > 0 and seq[i,j] >= 0:
                # print(seq[i,j])
                # if seq[i,j] not in weight4ind:
                #     print(seq[i,j])
                weight[i,j] = weight4ind[seq[i,j]]
    weight = np.asarray(weight, dtype='float32')
    return weight


def listFiles(dir):
    '''
    Takes a directory, dir, and returns a list of files, fileLst, that we will iterate over
    '''
    fileLst = [f for f in listdir(dir) if isfile(join(dir, f)) and not f.startswith('.')]
    return(fileLst)

def getSentences(ds, q):
    '''
    Given:
    :ds, directory where the files we will be looking at live (makes data_dir below)
    :q: 1 if there are questions as part of the dataset
    Return:
    :sents: dictionary where sentences['3026_b0_unusual.txt'] = a list where each element is one sentence of the text in that file
    :parts: list of files that we'll be looking at (Note: '3026_b0_unusual_q.txt' is not in this list)
    '''
    data_dir = 'data/' + ds + '_cleaned_tokenized'

    sents = {}
    parts = []
    filesInDir = listFiles(data_dir)
    for fileName in filesInDir:
        if '_q' not in fileName:
            parts.append(fileName)
        fsents = []
        with open(data_dir + '/' + fileName, 'r') as f:
            lines = f.readlines()
            for line in lines:
                fsents.append(line.strip())
        sents[fileName] = fsents
        if q == 0:
            sents[question(fileName)] = []
    return(sents, parts)

def question(part):
    '''
    Given '3026_b0_unusual.txt', return '3026_b0_unusual_q0.txt'
    '''
    components = part.split('.')
    return(components[0] + '_q.txt')

def getElmoEmbedding(sentences, layer):
    elmo = ElmoEmbedder()
    embedding = []
    for sent in sentences:
        if layer == 'all':
            v = np.mean(elmo.embed_sentence(sent.split()), axis = 0)
        else:
            v = elmo.embed_sentence(sent.split())[int(layer)]
        embedding.append(v)
    return(np.array(embedding))

def getPOSfromtuples(tpl):
    '''
    Get the following format: [POS_of_can, POS_of_you, POS_of_take, POS_of_out]
    From this format: [('Can', POS_of_can), ('you', POS_of_you), ('take', POS_of_take), ('out', POS_of_out),...]
    tpl: initial format, output from nltk
    '''
    lstofpos = []
    for ii in range(0, len(tpl)):
        lstofpos.append(tpl[ii][1])
    return(lstofpos)

def POStoYN(lst):
    # Define function vs. content words
    function = ['SYM', 'CC', 'CD', 'DT', 'EX', 'IN', 'LS', 'MD', 'PDT', 'POS', 'PRP', 'PRP$', 'RP', 'TO', 'UH', 'WDT', 'WP', 'WP$', 'WRB', '.', ',', ':', ')', '(', '""', "''", "'", '``', '$']
    content = ['FW', 'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    newlst = []     # output list: to be filled with 0 for function words, 1 for content words
    for item in lst:
        if item in function:
            newlst.append(0.)
        elif item in content:
            newlst.append(1.)
        else:
            newlst.append(0.)
            print("WARNING: POS " + str(item) + " is not in the yes/no list!")
    # If everything is a function word, then we do include the sentence in the analysis
    if np.count_nonzero(newlst) == 0:
        newlst = np.ones_like(newlst)
    return(newlst)

def getPOSWeights(sentences, x):
    '''
    Given the words in a sentence and the corresponding x matrix, return a matrix that has 0/1 whether the word is a content word
    '''
    # Create matrix w_pos that has the same shape as x
    w_pos = np.zeros_like(x)
    for ii in range(0, len(sentences)):
        # Tokenize sentences
        tokens = sentences[ii].split()
        # POS tag them (use nltk)
        pos_tpl = pos_tag(tokens)
        # pos_tag returns the following format: [('Can', POS_of_can), ('you', POS_of_you), ('take', POS_of_take), ('out', POS_of_out),...]
        # getPOSfromtuples() turns this into [POS_of_can, POS_of_you, POS_of_take, POS_of_out]
        lst_of_pos = getPOSfromtuples(pos_tpl)
        # POStoYN turns this into a list of 0/1s depending on function vs. content word status
        func_cont = POStoYN(lst_of_pos)
        w_pos[ii][:len(lst_of_pos)] = func_cont
    return(w_pos)

def write2file(out, model, sentembedding, dataset, shuffle):
    if len(out) == 0:
        return
    for ii in range(0, len(out)):
        split_part = out[ii][0].split('_')
        partID = split_part[0]
        if partID[0] in ['1', '3']:
            grp = 'chr'
        else:
            grp = 'hc'
        partTime = split_part[1]
        partQuestion = split_part[2].split('.')[0]
        outscores = out[ii][1:]
        out[ii] = [partID, partTime, grp, partQuestion] + outscores
    fn = 'results/' + dataset + '/' + dataset + '_' + model + '_' + sentembedding + '.csv'
    if shuffle: 
        fn = 'results/' + dataset + '/' + dataset + '_' + model + '_' + sentembedding + '_randomwords.csv'
    # print(fn)
    df = pd.DataFrame(out, columns=['Participant', 'Time', 'Group', 'Question','Coherence', 'Derailment', 'Iteration'])
    df.to_csv(fn, index=False)

def randomizewords(x, m, words):
    maxidx = len(words)
    x = np.random.randint(0,maxidx,x.shape)
    # Multiply by m to get rid of indices where there are no words
    x = np.multiply(x, m).astype('int32')
    return x

def getWordFreqs(x, w, ind2freq, part, n):
    out = []
    partid = part.split('_')[0]
    time = part.split('_')[1]
    qtype = part.split('_')[2].split('.')[0]
    for ii in range(n, x.shape[0]):
        for jj in range(0, x.shape[1]):
            if w[ii][jj] > 0:
                out.append([partid, time, qtype, ii-n, jj, ind2freq[x[ii][jj]]])
    return(out)