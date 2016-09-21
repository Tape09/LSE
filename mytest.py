from __future__ import print_function
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
import numpy as np
import Quandl


mydata = Quandl.get("CHRIS/CME_EH1", authtoken="-Mtn79XJPFoNyHWdyjfx", returns="numpy")
dates_data = mydata['Date']
settle_data = mydata['Settle']

# plt.plot(dates_data,settle_data)
# plt.savefig('foo1.png', bbox_inches='tight')
# exit()

'''Example script showing how to use stateful RNNs
to model long sequences efficiently.
'''

hidden_layer_size = 20;
# since we are using stateful rnn tsteps can be set to 1
tsteps = 1
batch_size = 1
epochs = 25
# number of elements ahead that are used to make the prediction
lahead = 5


def gen_cosine_amp(amp=100, period=25, x0=0, xn=50000, step=1, k=0.0001):
    """Generates an absolute cosine time series with the amplitude
    exponentially decreasing

    Arguments:
        amp: amplitude of the cosine function
        period: period of the cosine function
        x0: initial x of the time series
        xn: final x of the time series
        step: step of the time series discretization
        k: exponential rate
    """
    cos = np.zeros(((xn - x0) * step, 1, 1))
    for i in range(len(cos)):
        idx = x0 + i * step
        cos[i, 0, 0] = amp * np.cos(idx / (2 * np.pi * period))
        cos[i, 0, 0] = cos[i, 0, 0] * np.exp(-k * idx)
    return cos


print('Generating Data')
cos = np.zeros((settle_data.shape[0],1,1))
cos[:,0,0] = settle_data
print('Input shape:', cos.shape)

expected_output = np.zeros((len(cos), 1))
for i in range(len(cos) - lahead):
    expected_output[i, 0] = np.mean(cos[i + 1:i + lahead + 1])

print('Output shape')
print(expected_output.shape)

print('Creating Model')
model = Sequential()
model.add(LSTM(hidden_layer_size,
               batch_input_shape=(batch_size, tsteps, 1),
               return_sequences=True,
               stateful=True))
model.add(LSTM(hidden_layer_size,
               # batch_input_shape=(batch_size, tsteps, 1),
               return_sequences=False,
               stateful=True))
model.add(Dense(1))
model.compile(loss='mse', optimizer='rmsprop')

print('Training')
for i in range(epochs):
    print('Epoch', i, '/', epochs)
    model.fit(cos,
              expected_output,
              batch_size=batch_size,
              verbose=1,
              nb_epoch=1,
              shuffle=False)
    model.reset_states()

print('Predicting')
predicted_output = model.predict(cos, batch_size=batch_size)

print('Ploting Results')
plt.subplot(2, 1, 1)
plt.plot(expected_output)
plt.title('Expected')
plt.subplot(2, 1, 2)
plt.plot(predicted_output)
plt.title('Predicted')
# plt.show()
plt.savefig('asdf.png', bbox_inches='tight')