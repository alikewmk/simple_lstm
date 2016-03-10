import os
import sys
import itertools
import numpy as np
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

"""
TODO: Change the logic here to generate sentence char by char
"""

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
