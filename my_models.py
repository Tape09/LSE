from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
import keras.backend as K
import numpy as np





class s_lstm(object): #{
	def __init__(self,layers,input_size = 2,output_size = 2, batch_size=1, seq_size=1): #{
		self.model = Sequential()
		
		self.n_layers = len(layers)
		self.batch_size = batch_size;
		self.seq_size = seq_size;
		self.input_size = input_size;
		self.output_size = output_size;
		
		if(self.n_layers < 1): #{
			raise ValueError('you done fucked up now! n_layers must be >= 1')
		elif(self.n_layers == 1):
			self.model.add(	LSTM(layers[0],
							batch_input_shape=(self.batch_size, seq_size, self.input_size),
							return_sequences=False,
							stateful=True))			
		else:
			self.model.add(	LSTM(layers[0],
							batch_input_shape=(self.batch_size, seq_size, self.input_size),
							return_sequences=True,
							stateful=True))

			for i in range(1,self.n_layers-1): #{
				self.model.add(LSTM(layers[i],
									return_sequences=True,
									stateful=True))
			#}

			self.model.add(	LSTM(layers[-1],
							return_sequences=False,
							stateful=True))		
		#}

		self.model.add(Dense(self.output_size))
		self.model.compile(loss='mse', optimizer='rmsprop')
	#}

		
	def reset_states(self):
		self.model.reset_states()
		
	def fit(self,x,y):
		self.model.fit(	x,
						y,
						batch_size=self.batch_size,
						nb_epoch=1,
						shuffle=False,
						verbose = 0)
					
	def predict(self,x):
		return self.model.predict(x,batch_size=self.batch_size)
		
	def static_predict(self,x):
		old_states = self.get_states()
		ret = self.predict(x)
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
		
















