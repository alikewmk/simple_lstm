class CharFeatureGenerator:

    def __init__(self, source_file):
        self.data = open(source_file, 'r').read()
        self.chars      = list(set(self.data))
        self.data_size  = len(self.data)
        self.vocab_size = len(self.chars)
        self.dict_chars = {ch:i for i, ch in enumerate(self.chars)}
        self.dict_idx   = {i:ch for i, ch in enumerate(self.chars)}

    def print_info(self):
        print 'data has %d chars, %d unique.' % (self.data_size, self.vocab_size)

    def generate_training_data(self, gram_num):
        inputs = []
        targets = []
        p = 0
        while p+gram_num+1 < self.data_size:
            inputs.append([self.dict_chars[ch] for ch in self.data[p:p+gram_num]])
            targets.append([self.dict_chars[ch] for ch in self.data[p+1:p+gram_num+1]])
            p += 1

        return inputs, targets

# TEST
if __name__ == '__main__':
    train_chars = CharFeatureGenerator('data/char/pg1342.txt');
    train_chars.print_info()
    inputs, targets = train_chars.generate_training_data(3)
