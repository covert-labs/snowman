# IMPORTS
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv1D, Input, Dense, concatenate
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Embedding 
from tensorflow.keras.layers import GlobalMaxPooling1D
import numpy as np
from snowman.model.util.utilities import DataPrep
from sklearn.metrics import roc_curve, auc
import random

def text_cnn(max_seq_index, max_seq_length):
	text_input = Input(shape = (max_seq_length,), name='text_input')
	x = Embedding(output_dim=15, 
			input_dim=max_seq_index, 
			input_length=max_seq_length)(text_input)

	conv_a = Conv1D(15,2, activation='relu')(x)
	conv_b = Conv1D(15,4, activation='relu')(x)
	conv_c = Conv1D(15,6, activation='relu')(x)

	pool_a = GlobalMaxPooling1D()(conv_a)
	pool_b = GlobalMaxPooling1D()(conv_b)
	pool_c = GlobalMaxPooling1D()(conv_c)

	flattened = concatenate(
		[pool_a, pool_b, pool_c])

	drop = Dropout(.2)(flattened)

	outputs = []
	for x in range(89): # main output + 88 DGA families
		dense = Dense(1)(drop)
		out = Activation("sigmoid")(dense)
		outputs.append(out)

	model = Model(inputs=text_input, outputs=outputs)

	model.compile(
		loss='binary_crossentropy',
		optimizer='rmsprop',
		metrics=['accuracy']
	)

	return model

'''
ref: https://groups.google.com/forum/#!topic/keras-users/cpXXz_qsCvA

output1 = Dense(M, activation='softmax')(x) # now you create an output layer for each of your K groups. And each output has M elements, out of which because of 'softmax' only 1 will be activated. (practically this is of course a distribution, but after sufficient training, this usually makes one element close to one and the other elements close to zero)
output2 = Dense(M, activation='softmax')(x)
output3 = Dense(M, activation='softmax')(x)
... #you have to fill in the remaining layers here, or better: use a for loop
outputK = Dense(M, activation='softmax')(x)

model = Model(input=inputs, output=[output1, output2, output3, ..., outputK])


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(inputData, [outputData1, outputData2, outputData3, ... outputDataK], nb_epochs=10, batch_size=64)
'''


