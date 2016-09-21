import MySQLdb
import numpy as np
import Quandl 
import random as rand
from happy_funcs import *



class market_sim(object): #{

	def __init__(self, n_datasets, gen_sines = False, buy_penalty = 0.01, time_decay_penalty = 0.001, wrong_action_penalty = 0.01, not_bought_val = -1): #{
		self.fdata = list()
		self.day = 0
		self.fdata_idx = 0
		self.value = 1.0
		
		self.time_decay_penalty = time_decay_penalty;
		self.wrong_action_penalty = wrong_action_penalty;
		self.buy_penalty = buy_penalty;
		self.not_bought_val = not_bought_val
		
		self.last_action = 0;   # 0: stay #### 1:buy / sell
		self.bought = False;
		self.buy_price = self.not_bought_val;
		
		if(not gen_sines): #{
			db=MySQLdb.connect(read_default_file="~/.my2.cnf")
			c=db.cursor()
			c.execute("select dataset_code from michor.CHRIS_cfutures where name like '%#1 %'")
			self.codelist = rand.sample(c.fetchall(),n_datasets)
			
			i = 0
			pBar(i, n_datasets, prefix = 'Downloading datasets', suffix = 'Complete', barLength = 50) 
			for code in self.codelist: #{
				mydata = Quandl.get("CHRIS/"+code[0], authtoken="-Mtn79XJPFoNyHWdyjfx", returns="numpy")
				if("Settle" in mydata.dtype.names):				
					self.fdata.append(np.array(mydata['Settle']) / np.random.choice(mydata['Settle'],1)[0])
				elif("Previous Settlement" in mydata.dtype.names):
					self.fdata.append(np.array(mydata["Previous Settlement"]) / np.random.choice(mydata["Previous Settlement"],1)[0])
				elif("Settlement Price" in mydata.dtype.names):
					self.fdata.append(np.array(mydata["Settlement Price"]) / np.random.choice(mydata["Settlement Price"],1)[0])  
				i += 1
				pBar(i, n_datasets, prefix = 'Downloading datasets', suffix = 'Complete', barLength = 50)
			#}
		else:
			period = 300;
			offset = 200;
			for i in range(n_datasets):
				x = np.array(list(range(1000)));
				y = np.sin(2*np.pi*(1/(period+np.random.normal(0,100)))*(x+np.random.normal(0,offset)))/2 + 1;
				self.fdata.append(y);
		#}
		self.n_datasets = len(self.fdata)
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
		
	def step(self, action, for_real = True): #{
		today_price = self.get_price_today();
		tomorrow_price = self.get_price_tomorrow();
		
		old_value = self.value;
		new_value = self.value;
		new_buy_price = self.buy_price;
		
		if(self.bought and action == 1):			
			new_value *= (today_price / self.buy_price);
			new_buy_price = self.not_bought_val;
			if(for_real):
				self.bought = False;
				self.buy_price = new_buy_price;
		elif(self.bought and action == 0):
			pass;
		elif((not self.bought) and action == 1):
			new_value *= (1-self.buy_penalty);
			new_buy_price = today_price;
			if(for_real):
				self.bought = True;
				self.buy_price = new_buy_price;
		else:
			new_value *= (1-self.time_decay_penalty);

			
		if(for_real):
			self.value = new_value;
			self.day += 1;

		return(new_value/old_value-1, tomorrow_price, new_buy_price)		
	#}

	def step2(self, action, for_real = True): #{
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
		elif((not self.bought) and action == 0):
			new_value *= (1-self.time_decay_penalty);
		else:
			new_value *= (1-self.wrong_action_penalty);

			
		if(for_real):
			self.value = new_value;
			self.day += 1;

		return(new_value/old_value-1, tomorrow_price, new_buy_price)		
	#}
	
#}
		















