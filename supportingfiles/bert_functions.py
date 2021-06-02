import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as pd

# Get list of files in directory, dir
def listFiles(dir):
    fileLst = [f for f in listdir(dir) if isfile(join(dir, f)) and not f.startswith('.')]
    return(fileLst)

# Get dictionary with filename as key, sentences as value from directory + q (= 0 if no question)
def getSentences(dir, q):
    sents = {}
    parts = []
    filesInDir = listFiles(dir)
    for fileName in filesInDir:
        if '_q' not in fileName:
            parts.append(fileName)
        fsents = []
        with open(dir + '/' + fileName, 'r') as f:
            lines = f.readlines()
            for line in lines:
                fsents.append(line.strip())
        sents[fileName] = fsents
        if q == 0:
            sents[question(fileName)] = []
    return(sents, parts)

# Convert filename to question version of the filename
def question(part):
    components = part.split('.')
    return(components[0] + '_q.txt')

# Output results to file
def write2file(out, dataset):
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
    fn = 'results/' + dataset + '.csv'
    print(fn)
    df = pd.DataFrame(out, columns=['Participant', 'Time', 'Group', 'Question', 'Avg_Logit_Yes', 'Avg_Logit_No', 'Avg_Prob_Yes', 'Min_Prob_Yes', 'NSent'])
    df.to_csv(fn, index=False)