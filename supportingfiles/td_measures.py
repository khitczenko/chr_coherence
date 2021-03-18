import numpy as np
from scipy.stats import linregress
from scipy.spatial import distance

def getCosineSimilarity(vA, vB):
    cos = np.dot(vA, vB) / (np.sqrt(np.dot(vA,vA)) * np.sqrt(np.dot(vB,vB)))
    return(cos)

def coherence_score(embeddings):
	values = []
	for ii in range(1, len(embeddings)):
		vA = embeddings[ii-1]
		vB = embeddings[ii]
		cos = getCosineSimilarity(vA, vB)
		values.append(cos)
	return(np.mean(values))

# startidx is 0 or 1
# 0: include question
# 1: do not include question and use first sentence of answer as source
def derailment_score(embeddings, embeddings_q, startidx):
	if startidx == 0:
		source = np.mean(embeddings_q, axis = 0)
	else:
		source = embeddings[0]
	values = []	# keep track of what the cosine similarity is
	
	sentN = []	# keep track of which sentence we're comparing
	for ii in range(startidx, len(embeddings)):
		vB = embeddings[ii]
		cos = getCosineSimilarity(source, vB)
		sentN.append(ii)
		values.append(cos)
	if len(values) == 1:
		return(np.mean(values), 999)
	else:
		return(np.mean(values), linregress(sentN, values)[0])

def get_scores(part, embedding, n):
	# Split participant's sentences into question and answer
	embedding_q = embedding[0:n]
	embedding_a = embedding[n:]
	# Calculate "coherence"
	coherence = coherence_score(embedding_a)
	# DERAILMENT - "tangentiality in the paper"
	# last argument of derailment indicates whether we use the question as the source or not
	use_ans_as_source = 1	# 0 - use question as source; 1 - use ans as source
	as_slope = 1			# 0 - get average cosine, 1 - run extra linear regression step
	derailment = derailment_score(embedding_a, embedding_q, use_ans_as_source)[as_slope]
	return([part, round(coherence,4), round(derailment, 4)])