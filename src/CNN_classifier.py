import mxnet as mx
import pandas as pd
import numpy as np
import numpy
import config
import pickle
import hyper_parameters
from feature_generator import build_vocab, build_input_data
import os
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "1"

#download data 
#https://archive.ics.uci.edu/ml/datasets/Sentence+Classification


#read in indexed utterances
x_train = np.load(config.CNN_train_features)
x_test = np.load(config.CNN_test_features)

#load labels into memory
y_train = np.load(config.CNN_train_labels)
y_test = np.load(config.CNN_test_labels)

sentence_size = x_train.shape[1]

print x_train.shape
print y_train.shape
print x_test.shape
print y_test.shape

#load mapping between intents and indexed labels
pickle_obj = open(config.int_to_word, "rb")
int_to_word = pickle.load(pickle_obj)
pickle_obj.close()
vocab_size = len(int_to_word)


#data iterator to read from numpy files
train_iter = mx.io.NDArrayIter(data={'data': x_train},
                               label={'softmax_label': y_train},
                               batch_size=hyper_parameters.batch_size)

val_iter = mx.io.NDArrayIter(data={'data': x_test},
                             label={'softmax_label': y_test},
                             batch_size=hyper_parameters.batch_size)


#print the shape of features and labels for sanity check
print(train_iter.provide_data)
print(train_iter.provide_label)
i = 1
for batch in train_iter:
    if i == 1:
        print batch.data[0].asnumpy()
        print batch.data[0].shape
        print batch.label[0].asnumpy()
        print batch.label[0].shape
        break

#train from scratch is no input model is specified

if config.model_input_prefix is None:

    # placeholder for input features (label is automatically applied)
    X = mx.sym.var('data')

    #WORD EMBEDDING LAYER IN THE NEURAL NETWORK (BUYAAAAAH IF THIS WORKS)##############################################
    embed_layer = mx.sym.Embedding(data=X,
                                   input_dim=vocab_size,
                                   output_dim=hyper_parameters.vectorsize,
                                   name='vocab_embed')

    # reshape embedded data for next layer
    conv_input = mx.sym.Reshape(data=embed_layer,
                                shape=(hyper_parameters.batch_size, 1, sentence_size, hyper_parameters.vectorsize))
    #####################################################################################################################

    #MANY CONVOLUTIONAL LAYERS###############################################################################
    pooled_outputs = []
    for i, filter_size in enumerate(hyper_parameters.filter_list):
        #convolutional layer with a kernel that slides over entire words resulting in a 1d output
        convi = mx.sym.Convolution(data=conv_input, kernel=(
            filter_size, config.vectorsize), num_filter=hyper_parameters.num_filter)
        acti = mx.sym.Activation(data=convi, act_type='tanh')
        #take the max value of the convolution, sliding 1 unit (stride) at a time
        pooli = mx.sym.Pooling(data=acti, pool_type='max', kernel=(
            sentence_size - filter_size + 1, 1), stride=(1, 1))
        pooled_outputs.append(pooli)

    # combine all pooled outputs (just concatenate them since they are 1d now)
    total_filters = hyper_parameters.num_filter * \
        len(hyper_parameters.filter_list)
    concat = mx.sym.Concat(*pooled_outputs, dim=1)

    # reshape for next layer (this depends on batch size, meaning train and test need same batch size)
    h_pool = mx.sym.Reshape(data=concat, target_shape=(
        hyper_parameters.batch_size, total_filters))
    #apply dropout to this layer
    h_drop = mx.sym.Dropout(
        data=h_pool, p=hyper_parameters.dropout, mode='training')
    ###########################################################################################################

    # FULLY CONNECTED LAYER
    # recieve pooling layer and map to a neuron per class
    fc2 = mx.sym.FullyConnected(data=h_drop, num_hidden=config.classes)
    # Softmax with cross entropy loss function
    mlp = mx.sym.SoftmaxOutput(data=fc2, name='softmax')

    # create a trainable module on CPU/GPU
    model = mx.mod.Module(symbol=mlp,
                          context=config.context,
                          data_names=['data'],
                          label_names=['softmax_label'])


#resume training a pre-trained model
else:

    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix="../results/" + config.model_input_prefix,
                                                           epoch=hyper_parameters.epochs)
    model = mx.mod.Module(symbol=sym,
                          context=config.context,
                          data_names=['data'],
                          label_names=['softmax_label'])
    model.bind(for_training=True,
               data_shapes=val_iter.provide_data,
               label_shapes=val_iter.provide_label)
    model.set_params(arg_params, aux_params)

#print network architecture
print type(mx.viz.plot_network)

import logging
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

# reset train/test iterators to the beginning
train_iter.reset()
val_iter.reset()

print "fitting model..."
import os
#make sure we don't optimize convolutional layer in this script
print os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"]

model.fit(train_iter,  # train data
          # if validation data specified, training shows test metric after each epoch
          eval_data=val_iter,
          eval_metric='acc',
          optimizer=hyper_parameters.optimizer[0],
          optimizer_params=hyper_parameters.optimizer[1],
          # output progress for each 100 data batches
          batch_end_callback=mx.callback.Speedometer(
              hyper_parameters.batch_size, 100),
          num_epoch=hyper_parameters.epochs)  # train for x dataset passes

#save trained model at final epoch
model.save_checkpoint(prefix="../results/" + config.model_output_prefix,
                      epoch=hyper_parameters.epochs,
                      save_optimizer_states=False)
