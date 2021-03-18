import numpy as np
from sklearn.decomposition import TruncatedSVD
np.set_printoptions(precision=3, linewidth=200)

def compute_pc(X,npc=1):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_

def remove_pc(X, npc=1):
    """
    Remove the projection on the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    pc = compute_pc(X, npc)
    if npc==1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX

def SIF(We, x, w, iselmo):
    """
    Compute the scores between pairs of sentences using weighted average + removing the projection on the first principal component
    :param We: We[i,:] is the vector for word i
    :param x: x[i, :] are the indices of the words in the i-th sentence
    :param w: w[i, :] are the weights for the words in the i-th sentence
    :param params.rmpc: if >0, remove the projections of the sentence embeddings to their first principal component
    :return: emb, emb[i, :] is the embedding for sentence i
    """
    params = 1
    emb = get_weighted_average(We, x, w, iselmo)
    if  params > 0:
        emb = remove_pc(emb, params)
    return emb

def get_vector_length(We, iselmo):
    """
    Given the list of word vectors, return how long the vectors are
    :param We: We[i,:] is the vector for word i
    :param iselmo: indicator for whether we're looking at ELMo, which has different structure
    """
    if iselmo:
        ncol = We[0].shape[1]
    else:
        ncol = len(We[0])
    return(ncol)

def get_weighted_average(We, x, w, iselmo):
    """
    Compute the weighted average vectors
    :param We: We[i,:] is the vector for word i
    :param x: x[i, :] are the indices of the words in sentence i
    :param w: w[i, :] are the weights for the words in sentence i
    :return: emb[i, :] are the weighted average vector for sentence i
    """
    n_samples = x.shape[0]
    ncol = get_vector_length(We, iselmo)
    emb = np.zeros((n_samples, ncol))
    for i in range(n_samples):
        if iselmo:
            wN = w[i,:len(We[i])]
            emb[i,:] = wN.dot(We[i]) / np.count_nonzero(wN)
        else:           
            if len(We.shape) > 1:
                emb[i,:] = w[i,:].dot(We[x[i,:],:]) / np.count_nonzero(w[i,:])
            else:
                word_embeddings = We[x[i,:]]
                avg_word_embeddings = np.zeros(len(word_embeddings[0]))
                for jj in range(len(word_embeddings)):
                    avg_word_embeddings += w[i,jj]*word_embeddings[jj]
                emb[i,:] = avg_word_embeddings / np.count_nonzero(w[i,:])

    return emb