import numpy as np
import random as rand;
from market_sim import *
from my_models import *
from happy_funcs import *
import time
import keras.backend as K
import matplotlib.pyplot as plt

n_all_datasets = 10;
p_train_datasets = 0.5;
gen_sines = True;


msim = market_sim(n_all_datasets,gen_sines=gen_sines);
n_all_datasets = msim.get_n_datasets();

print("DATA SETS DOWNLOADED: "+str(n_all_datasets))

n_train_datasets = int(n_all_datasets*p_train_datasets);

verbose = 1; # 0:nothing, 1:bar, 2:debug

n_epochs = 100 # whole dataset epochs

gamma = 0.95;
eps_start = 1;
eps_end = 0.01;
n_epochs_to_eps_end = int(n_epochs/2)
eps_list = list(np.linspace(eps_start,eps_end,n_epochs_to_eps_end))
p_stay = 0.8;

n_states = 8;
n_layers = 8;
n_inputs = 2;
n_outputs = 3;

network_shape = [n_states]*n_layers;
model = s_lstm(network_shape,input_size = n_inputs, output_size = n_outputs);

dataset_idxs = list(range(n_all_datasets))
rand.shuffle(dataset_idxs);

train_idxs = dataset_idxs[:n_train_datasets];
test_idxs = dataset_idxs[n_train_datasets:];

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
		for day in range(n_days-1): #{
			world_state = np.array([[[msim.get_price_today(), msim.get_buy_price()]]])
			
			model_state = model.get_states();
			
			Q_zero = model.predict(world_state);
			if(rand.random() < eps): #random action
				if(rand.random() < p_stay):
					action = 0;
				else:
					action = rand.sample(list(range(1,n_outputs)),1)[0];
			else: #best estimate action
				action = np.argmax(Q_zero[0]);
			
			reward, new_price, buy_price = msim.step2(action);
			
			world_state_future = np.array([[[new_price,buy_price]]])
			Q_future = model.predict(world_state_future)
			
			maxQ = np.max(Q_future[0]);
			
			y = Q_zero;
			y[0,action] = reward + gamma * maxQ;
			model.set_states(model_state);
			model.fit(world_state,y);		
			
			if(verbose == 2 and day % 100 == 0):
				print("World_state: ",world_state)
				print("Q: ", Q_zero)
				print("")
			
		#}
		temp_rewards.append(msim.get_value());
		i += 1;
		if(verbose == 1):
			pBar(i, n_train_datasets, prefix = 'Training Datasets', suffix = 'Complete', barLength = 50)
	#}	
	epoch_rewards.append(np.mean(temp_rewards));
	print("Final Mean Reward: "+str(epoch_rewards[-1]))
#}


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
	for day in range(n_days-1): #{
		world_state = np.array([[[msim.get_price_today(), msim.get_buy_price()]]])
		Q = model.predict(world_state);
		action = np.argmax(Q_zero[0]);
		msim.step2(action);	
	#}
	final_rewards.append(msim.get_value());
#}

print("Test Rewards: ");
print(final_rewards);

print();

print("Mean: ");
print(np.mean(final_rewards));






























