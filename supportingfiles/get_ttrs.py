from lexical_diversity import lex_div as ld
import csv
import sys
sys.path.append('supportingfiles')
from data_io import listFiles

def getTTRs(file_dir, out_file):
	out = [['Participant', 'Time', 'Group', 'Question', 'MATTR', 'TotalWordCount', 'nSent', 'wordspersent']]

	# print(listFiles(file_dir))
	for f in listFiles(file_dir):
		splitfile = f.split('_')
		if '_q' in f:
			continue

		text = []
		with open(file_dir + f, 'r') as fo:
			lines = fo.readlines()
			for line in lines:
				text.append(line.strip())
			nsent = len(lines)
		text = ' '.join(text)
		
		tok = text.split()

		partid = splitfile[0]
		group = 'chr'
		if partid.startswith('4'):
			group = 'hc'
		time = splitfile[1]
		qtype = splitfile[2].split('.')[0]
		out.append([partid, time, group, qtype, ld.mattr(tok), len(tok), nsent, len(tok)/nsent])

	with open(out_file, "w", newline="\n") as f:
		writer = csv.writer(f)
		writer.writerows(out)

in_directory = "/Users/khit/Box Sync/Postdoc/Research/narratives_clean/data/nar_mainqonlynofillers_cleaned_tokenized/"
out_file = "/Users/khit/Box Sync/Postdoc/Research/narratives_clean/symps+demo/ttr_mainqonlynofillers.csv"
getTTRs(in_directory, out_file)