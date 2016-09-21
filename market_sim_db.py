import numpy as np
import random as rand
from happy_funcs import *
import sqlite3
import numpy.random as rng


class market_sim(object): #{

	def __init__(self, db_cursor, n_datasets, batch_size = 1, seq_size = 1, gen_sines = False, buy_penalty = 0.01, time_decay_penalty = 0.001, wrong_action_penalty = 0.01, not_bought_val = -1, ts_limit = 500): #{
		self.c = db_cursor;
		
		self.fdata = list()
		
		self.batch_size = batch_size;
		self.seq_size = seq_size;
		
		self.day = 0;
		self.set_idx = 0;
		self.batch_idx = 0;
		
		self.value = [1.0] * batch_size;		
		
		self.time_decay_penalty = time_decay_penalty;
		self.wrong_action_penalty = wrong_action_penalty;
		self.buy_penalty = buy_penalty;
		self.not_bought_val = not_bought_val
		
		self.bought = [False] * batch_size;
		self.buy_price = np.ones((batch_size,seq_size)) * self.not_bought_val;	
		
		data_list = [];
		
		if(not gen_sines): #{			
			q = "select distinct q_code from lse_data group by q_code having count(q_code)>?"
			self.codelist = self.c.execute(q,[ts_limit]).fetchall()
			self.codelist = [yy[0] for yy in self.codelist]
			
			if(n_datasets < 0 or n_datasets > len(self.codelist)):
				self.n_datasets = len(self.codelist);
			else:
				self.n_datasets = n_datasets;
				

			rand.shuffle(self.codelist);
			self.codelist = self.codelist[0:n_datasets]
			
			i = 0
			pBar(i, self.n_datasets, prefix = 'Downloading datasets', suffix = 'Complete', barLength = 50) 
			for code in self.codelist: #{
				q = "select price from lse_data where q_code=? order by date_ asc;"
				timeseries = self.c.execute(q,[code]).fetchall()
				y = np.array([yy[0] for yy in timeseries]);
				y = y.astype(np.unicode_)
				y = np.genfromtxt(map(lambda x: x.encode('UTF-8'),y),filling_values=np.nan) #this is needed because of reasons

				if(np.sum(np.isnan(y)) > 0): #{                #interpolate nans
					idxs = np.arange(y.size)
					idxs_non_null = np.logical_not(np.isnan(y));
					y = np.interp(idxs,idxs[idxs_non_null],y[idxs_non_null]);
				#}
				
				data_list.append(y / np.mean(y));				 
				i += 1;
				pBar(i, self.n_datasets, prefix = 'Downloading datasets', suffix = 'Complete', barLength = 50)
			#}
			print("\n")
		else:
			for i in range(n_datasets):
				random_freq = 0.25*(2*rng.rand() + 1)
				random_len = int(ts_limit + 0.1 * ts_limit * rng.rand() - 0.05 * ts_limit)
				random_phase = rng.rand()*2*np.pi
				x = np.array([0.01*j for j in range(random_len)])
				data_list.append(np.sin(2*np.pi*random_freq*x + random_phase).tolist()) 
		#}
		
		#populate self.fdata
		temp = fix_data(data_list,batch_size,seq_size);
		self.fdata = cut_data_to_batch_sets(temp,batch_size,seq_size);		
		
		self.n_datasets = len(data_list)
		self.n_sets = self.fdata.shape[0];
		self.n_batches_per_set = self.fdata.shape[1];
		self.batch_size = batch_size;
		self.seq_size = seq_size;
	#}
		
	def get_n_datasets(self):
		return(self.n_datasets);
		
	def get_value(self):
		return(self.value);
		
	def get_price_today(self):
		return(self.get_price(self.day))	
		
	def get_price_tomorrow(self):
		return(self.get_price(self.day+1))
		
	def get_price(self,day):
		return(self.fdata[self.fdata_idx][day])
		
	def get_fdata(self):
		return(self.fdata[self.fdata_idx]);
		
	def get_len(self):
		return(len(self.fdata[self.fdata_idx]))
		
	def get_day(self):
		return(self.day);
		
	def set_dset(self,idx):
		self.value = 1.0;
		self.day = 0;
		self.fdata_idx = idx;
		self.bought = False;
		self.buy_price = self.not_bought_val;
		return(self.fdata_idx)
		
	def get_dset(self):
		return(self.fdata_idx);
		
	def get_buy_price(self):
		return(self.buy_price);
		
	def has_bought(self):
		return(self.bought);
		
	def int_bought(self):
		return(1 if self.bought else -1);

		
	def batch_step(self, action, for_real=True):
		
		
		
	def step(self, action, for_real = True): #{
		today_price = self.get_price_today();
		tomorrow_price = self.get_price_tomorrow();
		
		old_value = self.value;
		new_value = self.value;
		new_buy_price = self.buy_price;
		
		if(self.bought and action == 2):		        #sell	
			new_value *= (today_price / self.buy_price);
			new_buy_price = self.not_bought_val;
			if(for_real):
				self.bought = False;
				self.buy_price = new_buy_price;
		elif(self.bought and action == 0):
			pass;
		elif((not self.bought) and action == 1):        #buy
			new_value *= (1-self.buy_penalty);
			new_buy_price = today_price;
			if(for_real):
				self.bought = True;
				self.buy_price = new_buy_price;
		elif((not self.bought) and action == 0):          #wait
			new_value *= (1-self.time_decay_penalty);
		else:
			new_value *= (1-self.wrong_action_penalty);

			
		if(for_real):
			self.value = new_value;
			self.day += 1;

		return(new_value/old_value-1, tomorrow_price, new_buy_price)		
	#}
	
#}
		















