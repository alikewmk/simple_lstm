import sys
import os
import numpy as np
from datetime import datetime
from simple_lstm import SimpleLSTM
from char_feature_generator import CharFeatureGenerator

_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.05'))
_NEPOCH = int(os.environ.get('NEPOCH', '100'))

def train_with_sgd(model, x_train, y_train,
                   learning_rate=0.001,
                   nepoch=10,
                   evaluate_loss_after=1):

    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        loss = model.calculate_loss(x_train, y_train)
        losses.append((num_examples_seen, loss))
        time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)

        # change learning rate here
        if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
            learning_rate = learning_rate * 0.5
            print "Setting learning rate to %f" % learning_rate

        """
        What is this for?
        """
        sys.stdout.flush()

        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(x_train[i], y_train[i], learning_rate)
            num_examples_seen += 1

"""
TODO: Change the logic here to generate sentence char by char
"""

def generate_sentence(model, train_chars, length=20):
    # randomly choose a start char
    # TODO: for each sentence, add start_char and end_char in the training data
    init_idx = np.random.choice(range(train_chars.vocab_size))
    new_sentence = [init_idx]
    while length > 0:
        next_word_probs = model.forward_propagation(new_sentence)
        print next_word_probs.shape
        length -= 1

        # generate via output o
        # http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.random.multinomial.html
        # get sample with largest probablity
        #samples = np.random.multinomial(1, next_word_probs[-1])
        #sampled_char_idx = np.argmax(samples)
        #new_sentence.append(sampled_char_idx)

        # generate via model prediction
        char_idxes = model.predict(new_sentence)
        print char_idxes
        char_idx = char_idxes[-1]
        new_sentence.append(char_idx)

    result_sentence = [train_chars.dict_idx[i] for i in new_sentence]

    return "".join(result_sentence)

# TEST
if __name__ == '__main__':

    train_chars = CharFeatureGenerator('data/char/pg1342.txt');
    train_chars.print_info()
    inputs, targets = train_chars.generate_training_data(30)

    model = SimpleLSTM(train_chars.vocab_size)

    train_with_sgd(model, inputs[:100], targets[:100], nepoch=_NEPOCH, learning_rate=_LEARNING_RATE)

    for i in range(10):
        print generate_sentence(model, train_chars)
