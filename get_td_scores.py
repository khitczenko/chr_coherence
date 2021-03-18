# python3 get_td_scores.py --MODEL --contentwords --ds --analyze/--freq/--shuffle

# TODO: in data_io, make sure that all of the word vectors are long enough (as done for glove)

# Adapted from github sent to me by Dan Iter, which is by someone else
import sys
import csv
import sent2vec
import torch
sys.path.append('supportingfiles')
import data_io, params, td_measures, sent_embeddings

# Select the number of iterations run. This is usually 1, but when shuffling words it is 1000.
nIter = 1
shuffle = 0
if len(sys.argv) == 5 and sys.argv[4] == '--shuffle':
	nIter = 1000
	shuffle = 1
print("Currently running with nIter = " + str(nIter))

# Deal with first command line argument
models = ['--lsa', '--word2vec', '--glove', '--elmo', '--sent2vec']
if sys.argv[1] not in models:
	print("Unrecognized model!")
	exit()
# model_index 0 for LSA, 1 for word2vec, etc.
model_index = models.index(sys.argv[1])

# Deal with second command line argument
# The following lists will hold the info that will be saved to file
mean_out, sif_out, tfidf_out, pos_out, s2v_out = [], [], [], [], []
# Set indices to 0/1 based on which sentence embeddings we want to run (can be more than one)
if sys.argv[2] == '--all':
	mean, sif, tfidf, pos, s2v = 1, 1, 1, 1, 0
else:
	sentembeddings = sys.argv[2][2:].split(',')
	mean, sif, tfidf, pos, s2v = 'mean' in sentembeddings, 'sif' in sentembeddings, 'tfidf' in sentembeddings, 'pos' in sentembeddings, 'sent2vec' in sentembeddings

# These contain the word to vector mappings
# Which one is used is chosen by first argument of the program (and model_index)
wordfiles = [
	'vectors/lsa.tasa.txt', #lsa
	'vectors/GoogleNews-vectors-negative300.bin', #word2vec
	'vectors/glove.840B.300d.txt', # glove
	'vectors/GoogleNews-vectors-negative300.bin'] #use word2vec vocab for elmo

# Get word to vector mapping for all but sent2vec
# words is a dictionary that has as its:
# 	- key: a word (str) in the word-to-vector mapping provided in the files above
#	- value: the index in We corresponding to that word's vector 
# We is a list of word vectors (we can use words to know which index to access)
# e.g. words['hello'] = 0, then we know that the 0th item in We is the word vector for 'hello'
if sys.argv[1] not in ['--sent2vec']:
	wordfile = wordfiles[model_index]
	# load word vectors
	(words, We) = data_io.getWordmap(wordfile, sys.argv[1])
	print("Wordmap complete")

####################################

# Load word weights (if TFIDF and SIF are one of the sentence embeddings)
# TODO: right now these are identical?
# word2weight['str'] is the weight for the word 'str'
# weight4ind[i] is the weight for the i-th word (where i is defined by words['str'])
if sif:
	word2weight_sif, weight4ind_sif = data_io.loadWordWeights('weightdata/enwiki_vocab_min200.txt', words)

if tfidf:
	word2weight_tfidf, weight4ind_tfidf = data_io.loadWordWeights('weightdata/enwiki_vocab_min200.txt', words)
print("SIF/TFIDF word weights complete")

####################################

## Get sentences
# First set up the data directory that we should be reading from
# ds = sys.argv[3]	# Which folder should we read from (in main text, this is nar_mainqonly)
# q = 0				# q is 0 if no question; 1 if there is a question
# if 'nar' in ds:		# nar: refers to the narratives dataset
# 	q = 1
# data_dir = 'data/' + ds + '_cleaned_tokenized'

# Then get sentences
# TODO: sentences is a dictionary here, but in what follows it's a list of strings, I believe
# sentences: dictionary where sentences['3026_b0_unusual.txt'] = a list where each element is one sentence of the text in that file
# parts: list of files that we'll be looking at (Note: '3026_b0_unusual_q.txt' is not in this list)
# ds: Which folder should we read from (in main text, this is nar_mainqonly)
# q is 0 if no question; 1 if there is a question
ds = sys.argv[3]
sentences, parts = data_io.getSentences(ds, q = 'nar' in ds)
# sentences, parts = data_io.getSentences(data_dir, q)
print("Sentences obtained")

# TODO: DELETE
# max_len = 0
# for part in parts: 
# 	sents = sentences[data_io.question(part)] + sentences[part]
# 	for sent in sents:
# 		input_ids = tokenizer.encode(sent, add_special_tokens = True)
# 		max_len = max(max_len, len(input_ids))
# print("Max sentence length: ", max_len)
# exit()

out = []

# if len(sys.argv) == 5 and sys.argv[4] == '--analyze':
# 	analyze = []
# 	wordfreq = []
# 	word2freq = data_io.getWordWeight('weightdata/enwiki_vocab_min200.txt', 'freq', 1.0)
# 	ind2freq = data_io.getWeight(words, word2freq)
# if len(sys.argv) == 5 and sys.argv[4] == '--freq':
# 	wordfreq = []
# 	word2freq = data_io.getWordWeight('weightdata/enwiki_vocab_min200.txt', 'freq', 1.0)
# 	ind2freq = data_io.getWeight(words, word2freq)
# 	# TODO: should I make the frequency of a word that hasn't occured lower?


for ii in range(0, nIter):
	print('Iteration: ' + str(ii))
	for part in parts:
		n = len(sentences[data_io.question(part)])
		
		## TODO: Check ELMo!!!
		# # For ELMo:
		# final arg determines which of three elmo layers is used (or if mean): 
		# options: '1', '2', '3', 'all' ('all' = mean)
		if sys.argv[1] == '--elmo':
			We = data_io.getElmoEmbedding(sentences[data_io.question(part)] + sentences[part], 'all')

		# # Get x, m, w (if applicable) - i.e. if not sent2vec
		# x is the array of word indices
		# m is the binary mask indicating whether there is a word in that location
		if sys.argv[1] not in ['--sent2vec']:
			x, m = data_io.sentences2idx(sentences[data_io.question(part)] + sentences[part], words)
			if len(sys.argv) == 5 and sys.argv[4] == '--shuffle':
				# x = data_io.randomizewords(x, m, len(words))
				x = data_io.randomizewords(x, m, words)
				# print("Words randomized")

			# Get mean embeddings + coherence, derailment
			if mean:
				# embedding = sent_embeddings.unweighted_mean(We, x, m, sys.argv[1] == '--elmo')
				# TODO: fix unweighted_mean function!!!
				# embedding = sent_embeddings.get_weighted_average(We, x, m, sys.argv[1] == '--elmo')
				embedding = sent_embeddings.get_weighted_average(We, x, m, sys.argv[1] == '--elmo')
				mean_out.append(td_measures.get_scores(part, embedding, n) + [ii])

			# Get only content word embeddings + coherence, derailment
			if pos:
				# TODO: check get_weighted_average!!!
				w_pos = data_io.getPOSWeights(sentences[data_io.question(part)] + sentences[part], x)
				embedding = sent_embeddings.get_weighted_average(We, x, w_pos, sys.argv[1] == '--elmo')
				pos_out.append(td_measures.get_scores(part, embedding, n) + [ii])
				# Check if we have an analyze option, meaning we want sentence by sentence predictions
				# if len(sys.argv) == 5 and sys.argv[4] == '--analyze':
				# 	analyze = analyze + td_measures.getSentbySent(part, embedding, sentences[part], n, x, w_pos, ind2freq)
				# if len(sys.argv) == 5 and sys.argv[4] == '--freq':
				# 	wordfreq = wordfreq + data_io.getWordFreqs(x, w_pos, ind2freq, part, n)

			# Get SIF embeddings + coherence, derailment
			if sif:
				# print(part)
				# for item in words:
				# 	idx = words[item]
				# 	if idx not in weight4ind_sif:
				# 		print(part)
				# 		print(words)
				# 		print(idx)
				# print("got here successfully")
				# wklejrkwerjwer

				# for i in range(x.shape[0]):
				# 	for j in range(x.shape[1]):
				# 		if m[i,j] > 0 and x[i,j] >= 0:
				# 			if x[i,j] not in weight4ind_sif:
				# 				print(x[i,j])
				# 				print(m[i,j])
				# 				print(x[i,j] in words.values())


				w_sif = data_io.seq2weight(x, m, weight4ind_sif) # get word weights	
				# embedding[i,:] is the embedding for sentence i in this part's passage
				embedding = sent_embeddings.SIF(We, x, w_sif, sys.argv[1] == '--elmo')
				sif_out.append(td_measures.get_scores(part, embedding, n) + [ii])

			# Get TF-IDF embeddings + coherence, derailment
			if tfidf:
				w_tfidf = data_io.seq2weight(x, m, weight4ind_tfidf) # get word weights
				embedding = sent_embeddings.get_weighted_average(We, x, w_tfidf, sys.argv[1] == '--elmo')
				tfidf_out.append(td_measures.get_scores(part, embedding, n) + [ii])

		# Get sentence embeddings for sent2vec
		else: 
			# Get sent2vec embedding
			s2v_model = sent2vec.Sent2vecModel()
			s2v_model.load_model('vectors/sent2vec.wiki.bigrams.bin')
			embedding = s2v_model.embed_sentences(sentences[data_io.question(part)] + sentences[part])
			s2v_out.append(td_measures.get_scores(part, embedding, n) + [ii])

# Write results out to individual files
allmodels = [('mean', mean_out), ('sif', sif_out), ('tfidf', tfidf_out), ('pos', pos_out), ('s2v', s2v_out)]
for m in allmodels:
	data_io.write2file(m[1], sys.argv[1][2:], m[0], ds, shuffle)

# if len(sys.argv) == 5 and sys.argv[4] == '--analyze':
# 	data_io.analysis2file(analyze, sys.argv[1][2:], 'pos', ds)
# if len(sys.argv) == 5 and sys.argv[4] == '--freq':
# 	data_io.wordfreq2file(wordfreq, sys.argv[1][2:], 'pos', ds)