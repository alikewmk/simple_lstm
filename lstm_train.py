import os
import sys
import itertools
import numpy as np
import numpy as np
import theano as theano
import theano.tensor as T
from six.moves import cPickle
from datetime import datetime
from simple_lstm import SimpleLSTM
from char_feature_generator import CharFeatureGenerator
from large_char_feature_generator import LargeCharFeatureGenerator

# prevent pickle stack over flow
sys.setrecursionlimit(100000)

_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.001'))
_NEPOCH = int(os.environ.get('NEPOCH', '1'))
_BATCH_SIZE = int(os.environ.get('NEPOCH', '100'))


"""
Change it to mini-batch training
"""
def train_with_sgd(model,
                   train_chars,
                   learning_rate=0.001,
                   nepoch=10,
                   evaluate_loss_after=1,
                   mini_batch_size=1000):

    losses = []
    num_examples_seen = 0

    # iterate through each epoch
    for epoch in range(nepoch):
        # iterate trough each mini batch
        # need to init after each iteration through whole dataset
        data_iterator = train_chars.generate_training_data()
        while 1:
            try:
                x_train, y_train = zip(*itertools.islice(data_iterator, 0, mini_batch_size))
                loss = model.calculate_loss(x_train, y_train)
                losses.append((num_examples_seen, loss))
                time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)

                # change learning rate here
                #if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                #    learning_rate = learning_rate * 0.5
                #    print "Setting learning rate to %f" % learning_rate

                sys.stdout.flush()

                for i in range(len(y_train)):
                    # One SGD step
                    model.sgd_step(x_train[i], y_train[i], learning_rate)
                    num_examples_seen += 1

            # iterator reaches the end here
            except ValueError:
                break

def generate_sentence(model, train_chars, length=20):
    # randomly choose a start char
    # TODO: for each sentence, add start_char and end_char in the training data
    init_idx = np.random.choice(range(train_chars.vocab_size))
    # use start sign * as the start of sentence
    new_sentence = [0]

    while length > 0:
        next_word_probs = model.forward_propagation(new_sentence)
        length -= 1

        # generate via output o
        # http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.random.multinomial.html
        # get sample with largest probablity
        samples = np.random.multinomial(1, next_word_probs[-1])
        sampled_char_idx = np.argmax(samples)
        new_sentence.append(sampled_char_idx)

        # generate via model prediction
        #char_idxes = model.predict(new_sentence)
        #print char_idxes
        #char_idx = char_idxes[-1]
        #new_sentence.append(char_idx)

    result_sentence = [train_chars.dict_idx[i] for i in new_sentence]

    return "".join(result_sentence)

def merge_model_params(models, epoch_num):
    """
    Merge the params of each model after current epoch, and store the params in a new model
    """
    model_num = len(models)

    # init vals need to merge
    first_model = models[0]
    sum_WiUi = np.zeros((first_model.hidden_num, first_model.concat_num))
    sum_WfUf = np.zeros((first_model.hidden_num, first_model.concat_num))
    sum_WoUo = np.zeros((first_model.hidden_num, first_model.concat_num))
    sum_WgUg = np.zeros((first_model.hidden_num, first_model.concat_num))
    sum_V    = np.zeros((first_model.vocab_num, first_model.hidden_num))
    sum_bi   = np.zeros((first_model.hidden_num))
    sum_bf   = np.zeros((first_model.hidden_num))
    sum_bo   = np.zeros((first_model.hidden_num))
    sum_bg   = np.zeros((first_model.hidden_num))

    # TODO: should I also average the bias here?
    for model in models:
        sum_WiUi += model.WiUi.eval()
        sum_WfUf += model.WfUf.eval()
        sum_WoUo += model.WoUo.eval()
        sum_WgUg += model.WgUg.eval()
        sum_V    += model.V.eval()
        sum_bi   += model.bi.eval()
        sum_bf   += model.bf.eval()
        sum_bo   += model.bo.eval()
        sum_bg   += model.bg.eval()

    mean_WiUi = sum_WiUi/model_num
    mean_WfUf = sum_WfUf/model_num
    mean_WoUo = sum_WoUo/model_num
    mean_WgUg = sum_WgUg/model_num
    mean_V    = sum_V/model_num
    mean_bi   = sum_bi/model_num
    mean_bf   = sum_bf/model_num
    mean_bo   = sum_bo/model_num
    mean_bg   = sum_bg/model_num

    for model in models:
        model.WiUi = theano.shared(name='WiUi', value=mean_WiUi.astype(theano.config.floatX))
        model.WfUf = theano.shared(name='WfUf', value=mean_WfUf.astype(theano.config.floatX))
        model.WoUo = theano.shared(name='WoUo', value=mean_WoUo.astype(theano.config.floatX))
        model.WgUg = theano.shared(name='WgUg', value=mean_WgUg.astype(theano.config.floatX))
        model.V    = theano.shared(name='V',  value=mean_V.astype(theano.config.floatX))
        model.bi   = theano.shared(name='bi', value=mean_bi.astype(theano.config.floatX))
        model.bf   = theano.shared(name='bf', value=mean_bf.astype(theano.config.floatX))
        model.bo   = theano.shared(name='bo', value=mean_bo.astype(theano.config.floatX))
        model.bg   = theano.shared(name='bg', value=mean_bg.astype(theano.config.floatX))

    first_model = models[0]

    with open('model_after_' + str(epoch_num) + "_epoch.save", 'wb') as f:
        cPickle.dump(first_model, f, protocol=cPickle.HIGHEST_PROTOCOL)

def timeit(method):
    """
    Decorator that calculate function running time
    """
    def calculate_time(*args, **kw):
        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()

        return ": ".join([result, str(end_time-start_time)]) + "\n"*2

    return calculate_time

# TEST
if __name__ == '__main__':

    train_chars = LargeCharFeatureGenerator('data/char/new_wsj_test.txt', 10);
    train_chars.print_info()

    if os.path.isfile("50000_model.save"):
        with open("50000_model.save",'rb') as f:
            model = cPickle.load(f)
    else:
        model = SimpleLSTM(train_chars.vocab_size)
        train_with_sgd(model,
                       train_chars,
                       nepoch=_NEPOCH,
                       learning_rate=_LEARNING_RATE,
                       mini_batch_size=_BATCH_SIZE)

        with open('50000_model.save', 'wb') as f:
            cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)

    for i in range(10):
        print generate_sentence(model, train_chars)
