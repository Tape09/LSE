from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
import keras.backend as K
import numpy as np
from happy_funcs import *




class s_lstm(object): #{
	def __init__(self,layers,input_size = 2,output_size = 2, batch_size=1, seq_size=1): #{
		self.model = Sequential()
		
		self.n_layers = len(layers)
		self.batch_size = batch_size;
		self.seq_size = seq_size;
		self.input_size = input_size;
		self.output_size = output_size;
		self.input_shape = (self.batch_size, seq_size, self.input_size)
		
		if(self.n_layers < 1): #{
			raise ValueError('you done fucked up now! n_layers must be >= 1')		
		else:
			self.model.add(	LSTM(layers[0],
							batch_input_shape=self.input_shape,
							return_sequences=True,
							stateful=True))

			for i in range(1,self.n_layers): #{
				self.model.add(LSTM(layers[i],
									return_sequences=True,
									stateful=True))
			#}	
		#}

		self.model.add(TimeDistributed(Dense(self.output_size)));
		self.model.compile(loss='mse', optimizer='rmsprop')
	#}

		
	def reset_states(self):
		self.model.reset_states()
		
	def train_on_batch(self,x,y):
		self.model.train_on_batch(	x,
						y)
					
	def predict_on_batch(self,x):
		return self.model.predict_on_batch(x)

	def test_on_batch(self,x,y):
		return self.model.test_on_batch(x,y)
		
	def static_test_on_batch(self,x,y):
		old_states = self.get_states()
		ret = self.test_on_batch(x,y)
		self.set_states(old_states)
		return ret
		
	def static_predict_on_batch(self,x):
		old_states = self.get_states()
		ret = self.predict_on_batch(x)
		self.set_states(old_states)
		return ret
		
	def get_states(self): #{
		n_layers = len(self.model.layers)-1;
		states = [None] * n_layers;
		
		for i in range(n_layers): #{
			temp = [None] * 2;
			for j in range(2):	#{
				temp[j] = K.get_value(self.model.layers[i].states[j])		
			#}	
			states[i] = temp;
		#}			
		return states;
	#}

	def set_states(self, states): #{
		n_layers = len(self.model.layers)-1;
		
		for i in range(n_layers): #{
			for j in range(2):	#{
				K.set_value(self.model.layers[i].states[j],states[i][j])		
			#}	
		#}
	#}
#}
		
















