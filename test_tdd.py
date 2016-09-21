import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Masking
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from keras.objectives import *
from keras.regularizers import *
import keras.backend as K
import numpy as np
import numpy.random as rng
from happy_funcs import *


def gen_sines(n,seq_len):
	out = [];
	for i in range(n):
		random_freq = 0.25*(2*rng.rand() + 1)
		random_len = int(seq_len + 0.1*seq_len*rng.rand() - 0.05*seq_len)
		random_phase = rng.rand()*2*np.pi
		x = np.array([0.01*j for j in range(random_len)])
		out.append(np.sin(2*np.pi*random_freq*x + random_phase).tolist()) 
	
	return out;
	
def fix_data(data,batch_size,seq_size):
	maxlen = np.max([len(s) for s in data]);
	maxlen += (seq_size - maxlen % seq_size) % seq_size;                    #make max sequence length divisible by seq_size		
	n_rows_to_add = (batch_size - len(data) % batch_size) % batch_size;		#make number of samples divisible by batch_size	
	data = pad_sequences(data,maxlen=maxlen,dtype='float')
	if n_rows_to_add > 0:
		data = np.vstack([data,np.zeros(n_rows_to_add,maxlen)])
	return data[:,:,None];
	
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
			batch_sets[iset,ibatch] = data[b_start:b_end,t_start:t_end]
		#}
	#}
	return batch_sets;
#}	
	
	
n_train_samples = 2000;
n_test_samples = 1000;
batch_size = 50;
seq_size = 50;
train_data = gen_sines(n_train_samples,500);
test_data = gen_sines(n_test_samples,500);

x_train = []
y_train = []
for i in range(len(train_data)):
	x_train.append(train_data[i][:-1]);
	y_train.append(train_data[i][1:]);
	
x_train = fix_data(x_train,batch_size,seq_size);
y_train = fix_data(y_train,batch_size,seq_size);
n_train_samples = x_train.shape[0];

x_test = []
y_test = []
for i in range(len(test_data)):
	x_test.append(test_data[i][:-1]);
	y_test.append(test_data[i][1:]);
	
x_test = fix_data(x_test,batch_size,seq_size);
y_test = fix_data(y_test,batch_size,seq_size);
n_test_samples = x_test.shape[0];


model = Sequential();
model.add(Masking(mask_value=0., batch_input_shape = (batch_size,seq_size,1)));
model.add(LSTM(4,return_sequences=True,stateful=True,W_regularizer=l2(0.01),U_regularizer=l2(0.01)));
model.add(LSTM(4,return_sequences=True,stateful=True,W_regularizer=l2(0.01),U_regularizer=l2(0.01)));
# model.add(LSTM(4,return_sequences=True,stateful=True,dropout_U=0.2,dropout_W=0.2,W_regularizer=l2(0.01),U_regularizer=l2(0.01)));
# model.add(LSTM(32,return_sequences=True,stateful=True,dropout_U=0.2,dropout_W=0.2,W_regularizer=l2(0.01),U_regularizer=l2(0.01)));
model.add(TimeDistributed(Dense(1)))
model.compile(loss='mse', optimizer='rmsprop')

x_train_sets = cut_data_to_batch_sets(x_train,batch_size,seq_size);
y_train_sets = cut_data_to_batch_sets(y_train,batch_size,seq_size);
x_test_sets = cut_data_to_batch_sets(x_test,batch_size,seq_size);
y_test_sets = cut_data_to_batch_sets(y_test,batch_size,seq_size);

n_sets = x_train_sets.shape[0];
n_batches_per_set = x_train_sets.shape[1];


epochs = 10;
for epoch in range(epochs): #{
	print("Epoch: "+str(1+epoch)+"/"+str(epochs));	
	err = 0;
	for iset in range(n_sets): #{	
		pBar(iset, n_sets-1, prefix = 'Training Datasets', suffix = 'Complete', barLength = 50) 
		for ibatch in range(n_batches_per_set): #{
			err += model.train_on_batch(x_train_sets[iset,ibatch],y_train_sets[iset,ibatch]);	
		#}
		model.reset_states();
	#}	
	err /= (n_sets*n_batches_per_set);
	print("Mean training error:",err);	
#}

y_test_pred = np.zeros(x_test.shape);

n_test_sets = x_test_sets.shape[0];
n_batches_per_test_set = x_test_sets.shape[1];

# err = 0;
# for iset in range(n_test_sets): #{	
	# pBar(iset, n_test_sets-1, prefix = 'Testing Datasets', suffix = 'Complete', barLength = 50) 
	# b_start = iset * batch_size;
	# b_end = (iset+1) * batch_size;
	# for ibatch in range(n_batches_per_test_set): #{
		# t_start = ibatch * seq_size;
		# t_end = (ibatch+1) * seq_size;
		# pred = model.predict_on_batch(x_test_sets[iset,ibatch]);
		# y_test_pred[b_start:b_end,t_start:t_end,:] = pred;  
		# err += np.mean(mean_squared_error(pred,y_test_sets[iset,ibatch]).eval());
	# #}
	# model.reset_states();
# #}	
# err /= (n_test_sets*n_batches_per_test_set);

# print("Mean_test_error:",err);	


err = 0;
for iset in range(n_sets): #{	
	pBar(iset, n_sets-1, prefix = 'Testing training Datasets', suffix = 'Complete', barLength = 50) 
	for ibatch in range(n_batches_per_set): #{
		err += model.test_on_batch(x_train_sets[iset,ibatch],y_train_sets[iset,ibatch]);	
	#}
	model.reset_states();
#}	
err /= (n_sets*n_batches_per_set);
print("Final Mean training error:",err);	

err = 0;
for iset in range(n_test_sets): #{	
	pBar(iset, n_test_sets-1, prefix = 'Testing test Datasets', suffix = 'Complete', barLength = 50) 
	for ibatch in range(n_batches_per_test_set): #{
		err += model.test_on_batch(x_test_sets[iset,ibatch],y_test_sets[iset,ibatch]);	
	#}
	model.reset_states();
#}	
err /= (n_test_sets*n_batches_per_test_set);
print("Final Mean testing error:",err);	



random_idx = rng.choice(y_test_pred.shape[0]);

plt.figure();
plt.plot(y_test[random_idx,:,0])
plt.plot(y_test_pred[random_idx,:,0])
plt.plot()
plt.savefig("test.png")
plt.close()


err = np.mean(np.square(y_test[random_idx,:,0] - y_test_pred[random_idx,:,0]));




































