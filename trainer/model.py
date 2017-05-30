import json
import os

from keras.preprocessing.text import Tokenizer
from tensorflow.python.lib.io import file_io
import csv
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Conv1D, Dropout, GlobalMaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers


def pad(x):
    return np.pad(x[:51], (0, 51 - len(x[:51])), 'constant')


def create_model(train_file, eval_file, job_dir, embedding_size, filters_count, train_batch_size, num_epochs,
                 eval_batch_size, dictionary_size):

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

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    model = Sequential()
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

    print('started compiling')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])


    print('started training')
    env = json.loads(os.environ.get('TF_CONFIG', '{}'))

    # Get the task information.
    task_info = env.get('task')

    trial = task_info.get('trial', '')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = ModelCheckpoint('model_%s.h5' % trial, save_best_only=True)

    model.fit(x_train, y_train, epochs=num_epochs, batch_size=train_batch_size,
              verbose=2, validation_data=(x_test, y_test),
              callbacks=[early_stopping, model_checkpoint])

    with file_io.FileIO('model_%s.h5' % trial, mode='r') as input_f:
        with file_io.FileIO(job_dir + '/model_%s.h5' % trial, mode='w+') as output_f:
            output_f.write(input_f.read())
