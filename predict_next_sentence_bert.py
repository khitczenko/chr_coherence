import torch
from transformers import *
import sys
sys.path.append('supportingfiles')
import bert_functions
import numpy as np

bert_model_to_use = 'bert-base-uncased'
# bert_model_to_use = 'bert-large-uncased'

### Resources:
# https://github.com/huggingface/transformers/issues/48

# Given two sentences, embed each of them
def embed_sentences(sent1, sent2):
	indexed_sent1 = tokenizer.encode(sent1, add_special_tokens = True)
	indexed_sent2 = tokenizer.encode(sent2, add_special_tokens = True)[1:]
	segment_ids = [0]*len(indexed_sent1) + [1]*len(indexed_sent2)
	indexed_tokens = indexed_sent1 + indexed_sent2
	tokens_tensor = torch.tensor([indexed_tokens])
	# A list which is the length of sent1 + sent2 (including CLS and SEP)
	# 0s for items in sent1 and 1s for item in sent2
	segments_tensor = torch.tensor([segment_ids])
	return(tokens_tensor, segments_tensor)

# ds = nar_mainqonlynofillers, nar_mainqonlyfillersin, etc.
ds = sys.argv[1]
# no question included 
q = 0
data_dir = '../narratives_clean/data/' + ds + '_cleaned_tokenized'

# Load sentences
sentences, parts = bert_functions.getSentences(data_dir, q)
print("Sentences obtained")

# Load BERT model
tokenizer = BertTokenizer.from_pretrained(bert_model_to_use)
model = BertForNextSentencePrediction.from_pretrained(bert_model_to_use)
model.eval()
print("BERT model loaded")

out = []

# Loop through files and get predictions of whether each pair of adjacent sentences follow e/o or not
for part in parts:
	avg_pred_yes = []
	avg_pred_no = []
	avg_pred_sm = []
	for ii in range(1, len(sentences[part])):
		# Embed each pair of adjacent sentences and get prediction
		tokens_tensor, segments_tensor = embed_sentences(sentences[part][ii-1], sentences[part][ii])
		predictions = model(tokens_tensor, token_type_ids=segments_tensor)
		pred_list = predictions[0].detach().numpy()
		softmax = torch.nn.Softmax(dim=1)
		predictions_sm = softmax(predictions[0])
		# Extract scores for whether these sentences are thought to follow e/o or not
		avg_pred_yes.append(pred_list[0][0])
		avg_pred_no.append(pred_list[0][1])
		avg_pred_sm.append(predictions_sm[0].detach().numpy()[0])
	# Average confidence that the sentences follow; average confidence that they don't follow; mean confidence that
	# they follow (soft-max), minimum confidence that they follow e/o (soft-max); # of sentences in passage 
	out.append([part, np.mean(avg_pred_yes), np.mean(avg_pred_no), np.mean(avg_pred_sm), min(avg_pred_sm), len(sentences[part])])

bert_functions.write2file(out, ds)