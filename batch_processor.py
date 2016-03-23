'''
Module consists of methods used in distributed training
!!! Remember to manually add:
data/batch/ -> training data separated into different files
models/results/ -> store the training result model of each epoch
models/training/ -> store the models currently training
logs/ store log of each training model
'''

from multiprocessing import Pool
from functools import partial
import os
import re
import sys
from six.moves import cPickle
from datetime import datetime

from lstm_train import *
from simple_lstm import SimpleLSTM
from char_feature_generator import CharFeatureGenerator
from large_char_feature_generator import LargeCharFeatureGenerator

_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.001'))
_NEPOCH = int(os.environ.get('NEPOCH', '1'))
_BATCH_SIZE = int(os.environ.get('NEPOCH', '100'))

def create_folder(folder):
    '''
    Create target dir for parsed files
    '''
    if not os.path.exists(folder):
        os.mkdir(folder)

def train(epoch_num, output_dir, *args):

    model_name = args[0][0]
    file       = args[0][1]
    log_name   = "logs/" + model_name + "_epoch_" + str(epoch_num) + ".log"
    model_name = output_dir + "training/" + model_name

    # direct stdout to log file
    log_file = open(log_name, 'a+')

    # TODO: gram_num here is a magic number!
    train_chars = LargeCharFeatureGenerator(file, 10);
    train_chars.print_info()

    if os.path.isfile(model_name):
        with open(model_name,'rb') as f:
            model = cPickle.load(f)
    else:
        model = SimpleLSTM(train_chars.vocab_size)

    train_with_sgd(model,
                   train_chars,
                   nepoch=_NEPOCH,
                   learning_rate=_LEARNING_RATE,
                   mini_batch_size=_BATCH_SIZE)

    with open(model_name, 'wb') as f:
        cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)

    for i in range(10):
        log_file.write(generate_sentence(model, train_chars))
        log_file.write("\n")

    log_file.close()

def multi_processing(process_num, epoch_num, input_dir, output_dir):
    '''
    The basic rule of batch number setting is:
    1. If processes needed is less than CPU number, set to process number
    2. If processes needed is more than CPU number, choose a batch number that is in [CPU_num, CPU_num*2)
       e.g: if we have 12 processes and 8 CPU, set number to 12 because 8 <= 12 < 8*2
            if we have 18 processes and 8 CPU, set number to 18/2=9 because  8 <= 9 < 8*2
    '''

    files = [input_dir + f for f in os.listdir(input_dir) if f.startswith('new_wsj')]
    model_names = ["model_" + str(i) for i in range(len(files))]
    train_process = partial(train, epoch_num, output_dir)
    pool = Pool(process_num)
    pool.map(train_process, zip(model_names, files))
    pool.close()
    pool.join()

    # TODO: There might have a more elegant way to avoid critical section
    models = []
    for model_name in model_names:
        model_name = output_dir + "training/" + model_name
        if os.path.isfile(model_name):
            with open(model_name,'rb') as f:
                model = cPickle.load(f)
        else:
            print model_name + " doesn't exist!"
            return
        models.append(model)
    merge_model_params(models, epoch_num, output_dir)

if __name__ == '__main__':
    for i in range(5):
        multi_processing(2, i+1, "data/batch_test/", "models/")
