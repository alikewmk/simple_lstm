import numpy as np
import theano as theano
import theano.tensor as T

class SimpleLSTM:

    """
    Should I also add bptt truncate
    """
    def __init__(self, vocab_num, hidden_num=100):
        concat_num = hidden_num + vocab_num
        self.vocab_num = vocab_num
        self.hidden_num = hidden_num
        self.concat_num = concat_num

        # weight init
        WiUi = np.random.uniform(-np.sqrt(1./concat_num), np.sqrt(1./concat_num), (hidden_num, concat_num))
        WfUf = np.random.uniform(-np.sqrt(1./concat_num), np.sqrt(1./concat_num), (hidden_num, concat_num))
        WoUo = np.random.uniform(-np.sqrt(1./concat_num), np.sqrt(1./concat_num), (hidden_num, concat_num))
        WgUg = np.random.uniform(-np.sqrt(1./concat_num), np.sqrt(1./concat_num), (hidden_num, concat_num))
        V    = np.random.uniform(-np.sqrt(1./hidden_num), np.sqrt(1./hidden_num), (vocab_num, hidden_num))
        self.WiUi = theano.shared(name='WiUi', value=WiUi.astype(theano.config.floatX))
        self.WfUf = theano.shared(name='WfUf', value=WfUf.astype(theano.config.floatX))
        self.WoUo = theano.shared(name='WoUo', value=WoUo.astype(theano.config.floatX))
        self.WgUg = theano.shared(name='WgUg', value=WgUg.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))

        # bias init
        bi = np.random.uniform(-np.sqrt(1./hidden_num), -np.sqrt(1./hidden_num), (hidden_num))
        bf = np.random.uniform(-np.sqrt(1./hidden_num), -np.sqrt(1./hidden_num), (hidden_num))
        bo = np.random.uniform(-np.sqrt(1./hidden_num), -np.sqrt(1./hidden_num), (hidden_num))
        bg = np.random.uniform(-np.sqrt(1./hidden_num), -np.sqrt(1./hidden_num), (hidden_num))
        self.bi = theano.shared(name='bi', value=bi.astype(theano.config.floatX))
        self.bf = theano.shared(name='bf', value=bf.astype(theano.config.floatX))
        self.bo = theano.shared(name='bo', value=bo.astype(theano.config.floatX))
        self.bg = theano.shared(name='bg', value=bg.astype(theano.config.floatX))

        self.build_model()

    def build_model(self):

        # assign to local
        WiUi = self.WiUi
        WfUf = self.WfUf
        WoUo = self.WoUo
        WgUg = self.WgUg
        V    = self.V
        bi   = self.bi
        bf   = self.bf
        bo   = self.bo
        bg   = self.bg


        x = T.ivector('x')
        y = T.ivector('y')

        def forward_prop_step(x_t, h_t_prev, c_t_prev,
                              WiUi, WfUf, WoUo, WgUg, V,
                              bi, bf, bo, bg):

            concate_input = T.concatenate([h_t_prev, x_t])
            i_t = T.nnet.sigmoid(WiUi.dot(concate_input) + bi)
            f_t = T.nnet.sigmoid(WfUf.dot(concate_input) + bf)
            o_t = T.nnet.sigmoid(WoUo.dot(concate_input) + bo)
            g_t = T.tanh(WgUg.dot(concate_input) + bg)

            c_t = c_t_prev * f_t + g_t * i_t
            h_t = T.tanh(c_t) * o_t
            y_hat = T.nnet.softmax(V.dot(h_t))

            """
            Is this wrong?
            """
            return [y_hat[0], h_t, c_t]

        one_hot_x = T.extra_ops.to_one_hot(x, self.vocab_num)
        [o, h, c], updates = theano.scan(
            forward_prop_step,
            sequences = one_hot_x,
            outputs_info=[
                # init o
                None,
                # init h
                dict(initial=T.zeros(self.hidden_num)),
                # init c
                dict(initial=T.zeros(self.hidden_num))
            ],
            non_sequences = [
                WiUi, WfUf, WoUo, WgUg, V,
                bi, bf, bo, bg
            ]
        )

        # Loss Function
        o_error = T.sum(T.nnet.categorical_crossentropy(o, y))

        dWiUi = T.grad(o_error, WiUi)
        dWfUf = T.grad(o_error, WfUf)
        dWoUo = T.grad(o_error, WoUo)
        dWgUg = T.grad(o_error, WgUg)
        dV    = T.grad(o_error, V)
        dbi   = T.grad(o_error, bi)
        dbf   = T.grad(o_error, bf)
        dbo   = T.grad(o_error, bo)
        dbg   = T.grad(o_error, bg)

        # used for SGD
        self.forward_propagation = theano.function([x], o)
        self.ce_error = theano.function([x, y], o_error)
        self.bptt = theano.function([x, y], [dWiUi, dWfUf, dWoUo, dWgUg, dV, dbi, dbf, dbo, dbg])

        # used for prediction
        prediction = T.argmax(o, axis=1)
        self.predict = theano.function([x], prediction)

        print "building sgd function..."
        learning_rate = T.scalar('learning_rate')
        self.sgd_step = theano.function(    [x, y, learning_rate],
                                            [],
                                            updates = [
                                                (self.WiUi, self.WiUi - learning_rate * dWiUi),
                                                (self.WfUf, self.WfUf - learning_rate * dWfUf),
                                                (self.WoUo, self.WoUo - learning_rate * dWoUo),
                                                (self.WgUg, self.WgUg - learning_rate * dWgUg),
                                                (self.V, self.V - learning_rate * dV),
                                                (self.bi, self.bi - learning_rate * dbi),
                                                (self.bf, self.bf - learning_rate * dbf),
                                                (self.bo, self.bo - learning_rate * dbo),
                                                (self.bg, self.bg - learning_rate * dbg)
                                            ]
                                        )
        print "done!"

    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x,y) for x,y in zip(X, Y)])

    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X,Y)/float(num_words)

