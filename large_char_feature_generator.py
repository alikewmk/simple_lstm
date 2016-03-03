import os
import string
import pickle

class LargeCharFeatureGenerator:

    def __init__(self, source_file, gram_num=10):
        self.source_file = source_file
        self.gram_num    = gram_num
        self.chars       = list(string.ascii_uppercase)
        self.vocab_size  = len(self.chars)
        self.dict_chars  = {ch:i for i, ch in enumerate(self.chars)}
        self.dict_idx    = {i:ch for i, ch in enumerate(self.chars)}
        self._generate_training_data()

    def print_info(self):
        print 'data has %d unique chars.' % (self.vocab_size)

    def line_corpus(self, filename):
        """
        Lazily read line corpus from a file `filename`
        """
        try:
            with open(filename, "r") as source:
                for line in source:
                    yield line.replace("\n", '').replace(" ", "")

        except IOError as error:
            exit(error)

    # maybe should change this part to yield!
    def _generate_training_data(self):
        inputs = []
        targets = []
        for line in self.line_corpus(self.source_file):
            p = 0
            while p+self.gram_num+1 < len(line):
                inputs.append([self.dict_chars[ch] for ch in line[p:p+self.gram_num]])
                targets.append([self.dict_chars[ch] for ch in line[p+1:p+self.gram_num+1]])
                p += 1

        self.inputs = inputs
        self.targets = targets

        return inputs, targets

# TEST
if __name__ == '__main__':

    gram_num = 10
    if os.path.isfile("data/" + str(gram_num) + '_gram_data'):
        with open("data/" + str(gram_num) + '_gram_data', 'rb') as f:
            train_chars = pickle.load(f)
    else:
        train_chars = LargeCharFeatureGenerator('data/char/new_wsj.txt');
        train_chars.print_info()
        with open("data/" + str(train_chars.gram_num) + '_gram_data','w') as f:
            pickle.dump(train_chars,f)

    for x in zip(train_chars.inputs, train_chars.targets):
        print x
