import json
import os

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.python.lib.io import file_io
import csv
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Conv1D, Dropout, GlobalMaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras import optimizers
from gensim.models.keyedvectors import KeyedVectors

def pad(x):
    return np.pad(x[:51], (0, 51 - len(x[:51])), 'constant')


def create_model(train_file, eval_file, job_dir, embedding_size, filters_count, train_batch_size, num_epochs,
                 eval_batch_size, dictionary_size, word_vectors_path):

    tokenizer = Tokenizer(dictionary_size)

    with file_io.FileIO(train_file, mode='r') as f:
        r = [l for l in csv.reader(f)]

        texts = [l[0] for l in r]
        tokenizer.fit_on_texts(texts)

        x_train = [np.array(pad(s)) for s in tokenizer.texts_to_sequences([l[0] for l in r])]
        y_train = [[l[1]] for l in r]

    with file_io.FileIO(eval_file, mode='r') as f:
        r = [l for l in csv.reader(f)]
        x_test = [np.array(pad(s)) for s in tokenizer.texts_to_sequences([l[0] for l in r])]
        y_test = [[l[1]] for l in r]

    if word_vectors_path:
        word_vectors = KeyedVectors.load_word2vec_format(word_vectors_path, binary=True)

        embedding_matrix = np.zeros((dictionary_size + 1, embedding_size))
        for w in sorted(tokenizer.word_counts.keys(), key=tokenizer.word_counts.get, reverse=True)[:dictionary_size]:
            try:
                embedding_vector = word_vectors.word_vec(w)
            except KeyError:
                continue
            embedding_matrix[tokenizer.word_index[w]] = np.array(embedding_vector)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    model = Sequential()
    if word_vectors_path:
        model.add(Embedding(dictionary_size + 1, 300, weights=[embedding_matrix]))
    else:
        model.add(Embedding(dictionary_size + 1, embedding_size))

    model.add(Dropout(0.5))

    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    model.add(Conv1D(filters_count,
                     3,
                     padding='valid',
                     activation='relu',
                     strides=1))
    # we use max pooling:
    model.add(GlobalMaxPooling1D())

    # We add a vanilla hidden layer:
    model.add(Dense(250))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])


    env = json.loads(os.environ.get('TF_CONFIG', '{}'))

    # Get the task information.
    task_info = env.get('task')
    if task_info:
        model_filename = 'model_%s.h5' % task_info.get('trial', '')
    else:
        model_filename = 'model.h5'

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    model_checkpoint = ModelCheckpoint(model_filename, save_best_only=True)
    tensorboard = TensorBoard(histogram_freq=1)

    model.fit(x_train, y_train, epochs=num_epochs, batch_size=train_batch_size,
              verbose=1, validation_data=(x_test, y_test),
              callbacks=[early_stopping, model_checkpoint, tensorboard])

    with file_io.FileIO(model_filename, mode='rb') as input_f:
        with file_io.FileIO(job_dir + '/' + model_filename, mode='bw+') as output_f:
            output_f.write(input_f.read())
