import os
import random
import string

class LargeCharFeatureGenerator:

    def __init__(self, source_file, gram_num=10):
        self.source_file = source_file
        self.gram_num    = gram_num
        # Mark * as the start of sentence
        # Mark $ as the end   of sentence
        self.chars       = ["*"] + list(string.ascii_uppercase) + ["$"]
        self.vocab_size  = len(self.chars)
        self.dict_chars  = {ch:i for i, ch in enumerate(self.chars)}
        self.dict_idx    = {i:ch for i, ch in enumerate(self.chars)}

    def print_info(self):
        print 'data has %d unique chars.' % (self.vocab_size)

    # remove blanks in data
    def line_corpus(self, filename):
        """
        Lazily read line corpus from a file `filename`
        """
        try:
            lines = open(filename).readlines()
            random.shuffle(lines)
            for line in lines:
                yield "".join(["*", line.replace("\n", '').replace(" ", ""), "$"])

        except IOError as error:
            exit(error)

    def generate_training_data(self):
        for line in self.line_corpus(self.source_file):
            p = 0
            while p+self.gram_num+1 < len(line):
                p += 1
                yield [self.dict_chars[ch] for ch in line[p:p+self.gram_num]], [self.dict_chars[ch] for ch in line[p+1:p+self.gram_num+1]]

# TEST
if __name__ == '__main__':

    train_chars = LargeCharFeatureGenerator('data/char/test.txt');
    train_chars.print_info()
    iterator = train_chars.generate_training_data()
    import itertools
    while 1:
        try:
            inputs, outputs = zip(*itertools.islice(iterator, 0, 100))
            print len(inputs)
        except ValueError:
            break

    print inputs

