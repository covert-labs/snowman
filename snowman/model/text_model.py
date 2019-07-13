# IMPORTS
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv1D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import GlobalMaxPooling1D
import numpy as np
from snowman.model.util.utilities import DataPrep
from snowman.model.text_cnn import text_cnn
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import random
import json
import os
import itertools

VERSION = "0.0.4-dga-vs-alexa"

pwd  = os.path.dirname(__file__)

MODEL_OUTPUT_FILEPATH = os.path.join(pwd, "../../fixtures/models/model_"+VERSION + "/")
MODEL_WEIGHTS_OUTPUT_FILEPATH = os.path.join(pwd,"../../fixtures/models/model_"+VERSION + "/weights")
MODEL_CONFIG_OUTPUT_FILEPATH = os.path.join(pwd,"../../fixtures/models/model_"+VERSION + "/config.json")

TRAINING_DATA_BLACKLIST_FILEPATH = os.path.join(pwd, "../../fixtures/datasets/dga_training.txt")
TRAINING_DATA_WHITELIST_FILEPATH = os.path.join(pwd, "../../fixtures/datasets/alexa_training.txt")

TESTING_DATA_BLACKLIST_FILEPATH = os.path.join(pwd, "../../fixtures/datasets/dga_testing.txt")
TESTING_DATA_WHITELIST_FILEPATH = os.path.join(pwd, "../../fixtures/datasets/alexa_testing.txt")

class TextModel(object):
	def __init__(self):
		self.version = VERSION
		self.max_sequence_length = None
		self.max_char_index = None
		self.net = None
		self.prep = DataPrep()

	def train(self):

		# data preparation
		print('Loading datasets ...')
		bl_strings = self.prep.load_url_file(TRAINING_DATA_BLACKLIST_FILEPATH)
		dga_labels = self.prep.load_labels(TRAINING_DATA_BLACKLIST_FILEPATH)
		wl_strings = self.prep.load_url_file(TRAINING_DATA_WHITELIST_FILEPATH)

		url_strings = bl_strings + wl_strings

		X = self.prep.to_one_hot_array(url_strings)
		Y = np.concatenate( [ np.ones(len(bl_strings)), np.zeros(len(wl_strings)) ])

		print('Creating DGA label Matrix ...')
		dga_families = ['bamital_dga','banjori_dga','bedep_dga','beebone_dga','blackhole_dga','bobax_dga','ccleaner_dga','chinad_dga','chir_dga','conficker_dga','corebot_dga','cryptolocker_dga','darkshell_dga','diamondfox_dga','dircrypt_dga','dnsbenchmark_dga','dnschanger_dga','downloader_dga','dyre_dga','ebury_dga','ekforward_dga','emotet_dga','feodo_dga','fobber_dga','gameover_dga','gameover_p2p','gozi_dga','goznym_dga','gspy_dga','hesperbot_dga','infy_dga','locky_dga','madmax_dga','makloader_dga','matsnu_dga','mirai_dga','modpack_dga','murofet_dga','murofetweekly_dga','necurs_dga','nymaim2_dga','nymaim_dga','oderoor_dga','omexo_dga','padcrypt_dga','pandabanker_dga','proslikefan_dga','pushdo_dga','pushdotid_dga','pykspa2_dga','pykspa2s_dga','pykspa_dga','qadars_dga','qakbot_dga','qhost_dga','ramdo_dga','ramnit_dga','ranbyus_dga','randomloader_dga','redyms_dga','rovnix_dga','shifu_dga','simda_dga','sisron_dga','sphinx_dga','suppobox_dga','sutra_dga','symmi_dga','szribi_dga','tempedreve_dga','tempedrevetdd_dga','tinba_dga','tinynuke_dga','tofsee_dga','torpig_dga','tsifiri_dga','ud2_dga','ud3_dga','ud4_dga','urlzone_dga','vawtrak_dga','vidro_dga','vidrotid_dga','virut_dga','volatilecedar_dga','wd_dga','xshellghost_dga','xxhex_dga']
		Y_dga = [Y]
		for dga in dga_families:
			Y_tmp = []
			for label in dga_labels:
				Y_tmp.append(1 if label == dga else 0)
			for x in range(len(wl_strings)):
				Y_tmp.append(0)
			Y_dga.append(np.array(Y_tmp))

		print('Building model ...')
		self.net = text_cnn(self.prep.max_index , self.prep.max_len)

		# model training
		train_test = train_test_split(X, *Y_dga, train_size=.5, shuffle=True)
		X_train, X_test, Y_train, Y_test = train_test[:4]
		dga_training_test = train_test[4:]
		
		all_Y_train = [Y_train]
		for idx in range(0, len(dga_training_test), 2):
			all_Y_train.append(dga_training_test[idx])

		self.net.fit(X_train, all_Y_train, batch_size=128, epochs=25)

		#model evaluation
		Y_pred = self.net.predict(X_test)
		print('Y_pred.shape = ',Y_pred.shape)

		# expecting this to fail ...
		try:
			fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
			auc_score = auc(fpr,tpr)
			print(f"\n AUC Score: {str(auc_score)}\n")
		except Exception as e:
			print(e)


	def save(self):
		print(f"Saving model under directory: { MODEL_OUTPUT_FILEPATH}")
		if not os.path.isdir(MODEL_OUTPUT_FILEPATH):
			os.mkdir(MODEL_OUTPUT_FILEPATH) 

		self.net.save(MODEL_WEIGHTS_OUTPUT_FILEPATH)
		model_configuration = {"max_sequence_length" : self.prep.max_len,
		"max_char_index": self.prep.max_index}
		with open(MODEL_OUTPUT_FILEPATH+"/config.json",'w+') as out_file:
			out_file.write(json.dumps(model_configuration))


	def load(self):
		print(f"Loading model config from: {MODEL_CONFIG_OUTPUT_FILEPATH}" )
		with open(MODEL_CONFIG_OUTPUT_FILEPATH, "r") as in_file:
			model_configuration = json.load(in_file)
		print(f"Loaded model config: {str(model_configuration)}")

		self.prep.max_len = model_configuration["max_sequence_length"]
		self.prep.max_index = model_configuration["max_char_index"]

		self.net = load_model(MODEL_WEIGHTS_OUTPUT_FILEPATH)

	def predict(self, input_string):
		transformed = self.prep.to_one_hot(
			input_string, self.prep.max_index, self.prep.max_len)
		score = self.net.predict(transformed)
		return score