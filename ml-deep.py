from keras.models import Sequential, Model, Input
from keras.layers import Dot, Dense, Dropout, Embedding, Reshape, Conv1D, MaxPooling1D
from keras.layers.merge import Concatenate
from keras.callbacks import ModelCheckpoint
from keras import metrics, utils, losses
from keras.utils.vis_utils import model_to_dot
from sklearn import metrics as sk_metrics, cross_validation
from IPython.display import SVG
import matplotlib.pyplot as plt
from pylab import plot
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

k = 128
data = pd.read_csv('./mldata/ratings.dat', sep='::', engine='python', names=['uid', 'sid', 'rating', 'time'])
n_user = data.uid.unique().shape[0]
n_sub = data.sid.unique().shape[0]
print n_user, n_sub

input1 = Input(shape=(1,))
model1 = Embedding(n_user, k, input_length=1)(input1)
model1 = Reshape((k,))(model1)
input2 = Input(shape=(1,))
model2 = Embedding(n_sub, k, input_length=1)(input2)
model2 = Reshape((k,))(model2)
model = Concatenate()([model1, model2])
model = Dropout(0.2)(model)
model = Dense(k, activation='relu')(model)
model = Dropout(0.5)(model)
model = Dense(int(k/4), activation='relu')(model)
model = Dropout(0.5)(model)
model = Dense(int(k/16), activation='relu')(model)
model = Dropout(0.5)(model)
output = Dense(1, activation='linear')(model)
model = Model([input1, input2], output)
model.compile(loss='mse', optimizer='adam', metrics=['mae', 'acc'])
SVG(model_to_dot(model).create(prog='dot', format='svg'))

train, test = cross_validation.train_test_split(data, test_size=0.1, random_state=1)
x_train = [train.uid, train.sid]
y_train = train.rating
x_test = [test.uid, test.sid]
y_test = test.rating
history = model.fit(x_train, y_train, batch_size=500, epochs=20, validation_data=(x_test, y_test))

save_path = './save/ml-deep/'
model.save(save_path + 'model.h5')
with open(save_path + 'history.pkl', 'wb') as file_history:
    pickle.dump(history.history, file_history)
pd.DataFrame(history.history, columns=['loss', 'val_loss']).head(20).transpose()
plot(history.history['loss'], label='loss')
plot(history.history['val_loss'], label='val_loss')
plot(history.history['val_mean_absolute_error'], label='val_mae')
plot(history.history['val_acc'], label='val_acc')
plt.legend()
plt.ylim(0, 3)
