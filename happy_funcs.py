import sys
from keras.preprocessing.sequence import pad_sequences
import numpy as np


# Print iterations progresss
def pBar (iteration, total, prefix = '', suffix = '', decimals = 2, barLength = 100):
	# """
	# Call in a loop to create terminal progress bar
	# @params:
		# iterations  - Required  : current iteration (Int)
		# total       - Required  : total iterations (Int)
		# prefix      - Optional  : prefix string (Str)
		# suffix      - Optional  : suffix string (Str)
	# """
	filledLength    = int(round(barLength * iteration / float(total)))
	percents        = round(100.00 * (iteration / float(total)), decimals)
	bar             = '#' * filledLength + '-' * (barLength - filledLength)
	sys.stdout.write('%s [%s] %s%s %s\r' % (prefix, bar, percents, '%', suffix)),
	sys.stdout.flush()
	if iteration == total:
		print("\n")

		
# prepad with zeros, and return in numpy format
def fix_data(data,batch_size,seq_size):
	maxlen = np.max([len(s) for s in data]);
	maxlen += (seq_size - maxlen % seq_size) % seq_size;                    #make max sequence length divisible by seq_size		
	n_rows_to_add = (batch_size - len(data) % batch_size) % batch_size;		#make number of samples divisible by batch_size	
	data = pad_sequences(data,maxlen=maxlen,dtype='float')
	if n_rows_to_add > 0:
		data = np.vstack([data,np.zeros(n_rows_to_add,maxlen)])
	return data[:,:,None];
	
# fixed data split into batch sets for ez input to keras
def cut_data_to_batch_sets(data,batch_size,seq_size): #{                           
	n_sets = data.shape[0] // batch_size;
	n_batches_per_set = data.shape[1] // seq_size;	
	
	batch_sets = np.zeros((n_sets,n_batches_per_set,batch_size,seq_size,1));	
	for iset in range(n_sets): #{
		b_start = iset * batch_size;
		b_end = (iset+1) * batch_size;
		for ibatch in range(n_batches_per_set): #{
			t_start = ibatch * seq_size;
			t_end = (ibatch+1) * seq_size;
			batch_sets[iset,ibatch] = data[b_start:b_end,t_start:t_end];
		#}
	#}
	return batch_sets;
#}	