from attention_decoder import AttentionDecoder
import numpy as np
from keras.models import Sequential
from keras.layers import GRU 
from random import randint

# generate a sequence of random integers
def generate_sequence(length, n_unique):
    return [randint(0, n_unique - 1) for _ in range(length)]


# one hot encode sequence
def encode_vector(sequence, n_unique):
    encoding = list()
    for value in sequence:
        vector = [0 for _ in range(n_unique)]
        vector[value] = 1
        encoding.append(vector)
    return np.array(encoding)


def decode_vector(encoded_seq):
    return [np.argmax(vector) for vector in encoded_seq]


# prepare data for the GRU
def get_pair(n_in, n_out, cardinality):
    # generate random sequence
    sequence_in = generate_sequence(n_in, cardinality)
    sequence_out = sequence_in[:n_out] + [0 for _ in range(n_in - n_out)]
    # one hot encode
    X = encode_vector(sequence_in, cardinality)
    y = encode_vector(sequence_out, cardinality)
    # reshape as 3D
    X = X.reshape((1, X.shape[0], X.shape[1]))
    y = y.reshape((1, y.shape[0], y.shape[1]))
    return X, y


def run():
    # configure problem
    features = 50
    in_step = 5
    out_step = 2

    # define model
    model = Sequential()
    model.add(GRU(150, input_shape=(in_step, features), return_sequences=True))
    model.add(AttentionDecoder(150, features))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # train GRU
    for epoch in range(5000):
        # generate new random sequence
        X, y = get_pair(in_step, out_step, features)
        # fit model for one epoch on this sequence
        model.fit(X, y, epochs=1, verbose=2)
    # evaluate GRU
    total, correct = 100, 0
    for _ in range(total):
        X, y = get_pair(in_step, out_step, features)
        yhat = model.predict(X, verbose=0)
        if np.array_equal(decode_vector(y[0]), decode_vector(yhat[0])):
            correct += 1
    print('Accuracy: %.2f%%' % (float(correct) / float(total) * 100.0))
    # spot check some examples
    for _ in range(10):
        X, y = get_pair(in_step, out_step, features)
        yhat = model.predict(X, verbose=0)
        print('Expected:', decode_vector(y[0]), 'Predicted', decode_vector(yhat[0]))


if __name__ == "__main__":
    run()
