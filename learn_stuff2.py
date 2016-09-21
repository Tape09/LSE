import matplotlib
matplotlib.use("Agg")

import numpy as np
import random as rand;
from market_sim_db import *
from my_models import *
from happy_funcs import *
from teacher_ai import *
import time
import keras.backend as K
import matplotlib.pyplot as plt
import h5py
import os
import sqlite3

if(len(sys.argv)<2):
	print("Not enough arguments.")
	sys.exit(-1)

out_folder = sys.argv[1]

if not os.path.exists(out_folder):
    os.makedirs(out_folder)

weights_file = out_folder + "/weights.h5"
codelist_file = out_folder + "/codelist.npz"
train_idxs_file = out_folder + "/train.npz"
test_idxs_file = out_folder + "/test.npz"


conn = sqlite3.connect("london/LSE_DB.db")
c = conn.cursor()

n_all_datasets = -1;
p_train_datasets = 0.80;
gen_sines = False;


msim = market_sim(c,n_all_datasets,gen_sines=gen_sines);
n_all_datasets = msim.get_n_datasets();

print("DATA SETS DOWNLOADED: "+str(n_all_datasets))

n_train_datasets = int(n_all_datasets*p_train_datasets);

verbose = 1; # 0:nothing, 1:bar, 2:debug

n_epochs = 100 # whole dataset epochs
gamma = 0.95;
eps_start = 1.0;
eps_end = 0.0;
n_epochs_to_eps_end = int(n_epochs*0.75)
eps_list = list(np.linspace(eps_start,eps_end,n_epochs_to_eps_end))
# p_stay = 0.8;

n_states = 8;
n_layers = 4;
n_inputs = 2;
n_outputs = 3;

network_shape = [n_states]*n_layers;
model = s_lstm(network_shape,input_size = n_inputs, output_size = n_outputs);

dataset_idxs = list(range(n_all_datasets))
rand.shuffle(dataset_idxs);

train_idxs = dataset_idxs[:n_train_datasets];
test_idxs = dataset_idxs[n_train_datasets:];

np.savez(codelist_file,msim.codelist)
np.savez(train_idxs_file,train_idxs)
np.savez(test_idxs_file,test_idxs)


print("n_layers:",n_layers)
print("n_states:",n_states)
print("n_all_datasets:",n_all_datasets)
print("n_train_datasets:",n_train_datasets)


t0 = time.time()
t0_cpu = time.clock()
epoch_rewards = []
for epoch in range(n_epochs): #{
	print("Epoch: "+str(epoch+1)+"/"+str(n_epochs))
	rand.shuffle(train_idxs);
	
	if(epoch<len(eps_list)):
		eps = eps_list[epoch];
	else:
		eps = eps_list[-1]
	
	temp_rewards = []
	i=0;
	if(verbose == 1):
		pBar(i, n_train_datasets, prefix = 'Training Datasets', suffix = 'Complete', barLength = 50) 
	for idx in train_idxs: #{
		msim.set_dset(idx);
		n_days = msim.get_len();
		model.reset_states();
		last_action = 0;
		teacher = teacher_ai(msim.get_fdata())
		for day in range(n_days-1): #{
			world_state = np.array([[[msim.get_price_today(), msim.get_buy_price()]]])  #world state is todays price and past buy price
			model_state = model.get_states();											#model state is internal state of lstm
			
			Q_zero = model.predict(world_state);										#predict future rewards for each action
			if(rand.random() < eps): #teacher action
				action = teacher.get_action(msim.get_day());
			else: #best estimate action
				action = np.argmax(Q_zero[0]);
			
			reward, new_price, buy_price = msim.step(action);							#make action
			world_state_future = np.array([[[new_price,buy_price]]])					#new state
			Q_future = model.predict(world_state_future)								#future rewards for t+1			
			maxQ = np.max(Q_future[0]);													#best future reward
			
			y = Q_zero;
			y[0,action] = reward + gamma * maxQ;										#update q value for previous action. immediate reward + gamme*future reward
			model.set_states(model_state);												#reset internal state
			model.fit(world_state,y);													#supervised training with world state and y
			
			last_action = action;
			
			if(verbose == 2 and day % 100 == 0):
				print("World_state: ",world_state)
				print("Q: ", Q_zero)
				print("Last_action: ", last_action)
			
		#}
		temp_rewards.append(msim.get_value());
		
		i += 1;
		if(verbose == 1):
			pBar(i, n_train_datasets, prefix = 'Training Datasets', suffix = 'Complete', barLength = 50)
	#}	

	model.model.save_weights(weights_file,overwrite=True);								#in case of bullshit - save periodically
	epoch_rewards.append(np.mean(temp_rewards));
	print("Final Mean Reward: "+str(epoch_rewards[-1]))
#}

model.model.save_weights(weights_file,overwrite=True);

print("=============TRAINING COMPLETE===============")
tdiff = time.time() - t0
tcpu_diff = time.clock() - t0_cpu
print("Time Taken: ", tdiff)
print("CPU time: ", tcpu_diff)
print()



print("TESTING...")


final_rewards = []
for idx in test_idxs: #{
	model.reset_states();
	msim.set_dset(idx);
	n_days = msim.get_len();
	last_action = 0;
	for day in range(n_days-1): #{
		world_state = np.array([[[msim.get_price_today(), msim.get_buy_price(),last_action]]])
		Q = model.predict(world_state);
		action = np.argmax(Q_zero[0]);
		msim.step(action);	
	#}
	final_rewards.append(msim.get_value());
#}

print("Test Rewards: ");
print(final_rewards);

print();

print("Mean: ");
print(np.mean(final_rewards));






























