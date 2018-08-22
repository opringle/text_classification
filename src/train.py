import pandas as pd
import numpy as np
import pickle

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

# Read in files
intent_to_index = load_obj('../data/ag_news_char/intent_to_index')
char_to_index = load_obj('../data/ag_news_char/char_to_index')
train_df = pd.read_pickle('../data/ag_news_char/train.pickle')
test_df = pd.read_pickle('../data/ag_news_char/test.pickle')

# Feature/label lists
X_train = train_df.X.tolist()
Y_train = train_df.Y.tolist()
X_test = test_df.X.tolist()
Y_test = test_df.Y.tolist()

print("\nDictionary mapping characters to integer indices:\n{}\n\nDictionary mapping labels to integer indices:\n{}".format(char_to_index, intent_to_index))
train_df.head()


import bisect
import random
import numpy as np
from mxnet.io import DataIter, DataBatch, DataDesc
from mxnet import ndarray
from sklearn.utils import shuffle


class BucketUtteranceIter(DataIter):
    """
    This iterator can handle variable length feature arrays
    """
    def __init__(self, utterances, intents, batch_size, buckets, data_pad=-1, label_pad=-1, data_name='utterance',
                 label_name='intent', dtype='float32'):
        """
        :param utterances: list of list of int
        :param intents: list of int
        """
        super(BucketUtteranceIter, self).__init__()
        buckets.sort()

        nslice = 0  # Keep track of how many utterances are sliced
        self.utterances = [[] for _ in buckets]
        self.intents = [[] for _ in buckets]
        self.indices = [[] for _ in buckets]

        for i, utt in enumerate(utterances):
            # Find the index of the smallest bucket that is larger than the sentence length
            buck_idx = bisect.bisect_left(buckets, len(utt))

            # Slice utterances that are too long to the largest bucket size
            if buck_idx == len(buckets):
                buck_idx = buck_idx - 1
                nslice += 1
                utt = utt[:buckets[buck_idx]]

            # Pad utterances that are too short for their bucket
            buff = np.full((buckets[buck_idx]), data_pad, dtype=dtype)
            buff[:len(utt)] = utt

            # Add data/label to bucket
            self.utterances[buck_idx].append(buff)
            self.intents[buck_idx].append(intents[i])
            self.indices[buck_idx].append(i)

        # Convert to list of array
        self.utterances = [np.asarray(i, dtype=dtype) for i in self.utterances]
        self.intents = [np.asarray(i, dtype=dtype) for i in self.intents]
        self.indices = [np.asarray(i, dtype=dtype) for i in self.indices]

        print("\nWarning, {0} utterances sliced to largest bucket size.".format(nslice)) if nslice > 0 else None
        print("Utterances per bucket: {}\nBucket sizes: {}".format([arr.shape[0] for arr in self.utterances], buckets))

        self.data_name = data_name
        self.label_name = label_name
        self.batch_size = batch_size
        self.buckets = buckets
        self.dtype = dtype
        self.data_pad = data_pad
        self.label_pad = label_pad
        self.default_bucket_key = max(buckets)
        self.layout = 'NT'

        self.provide_data = [DataDesc(name=self.data_name,
                                      shape=(self.batch_size, self.default_bucket_key),
                                      layout=self.layout)]
        self.provide_label = [DataDesc(name=self.label_name,
                                       shape=(self.batch_size, ),
                                       layout=self.layout)]

        # create empty list to store batch index values
        self.idx = []
        for i, buck in enumerate(self.utterances):
            self.idx.extend([(i, j) for j in range(0, len(buck) - batch_size + 1, batch_size)])
        self.curr_idx = 0
        self.reset()

    def reset(self):
        """
        Resets the iterator to the beginning of the data.
        """
        self.curr_idx = 0
        # shuffle data in each bucket
        random.shuffle(self.idx)
        for i, buck in enumerate(self.utterances):
            self.indices[i], self.utterances[i], self.intents[i] = shuffle(self.indices[i],
                                                                           self.utterances[i],
                                                                           self.intents[i])
        self.ndindex = []
        self.ndsent = []
        self.ndlabel = []

        # append the lists with an array
        for i, buck in enumerate(self.utterances):
            self.ndindex.append(ndarray.array(self.indices[i], dtype=self.dtype))
            self.ndsent.append(ndarray.array(self.utterances[i], dtype=self.dtype))
            self.ndlabel.append(ndarray.array(self.intents[i], dtype=self.dtype))

    def next(self):
        """
        Returns the next batch of data.
        """
        if self.curr_idx == len(self.idx):
            raise StopIteration
        # i = batches index, j = starting record
        i, j = self.idx[self.curr_idx]
        self.curr_idx += 1

        indices = self.ndindex[i][j:j + self.batch_size]
        utterances = self.ndsent[i][j:j + self.batch_size]
        intents = self.ndlabel[i][j:j + self.batch_size]

        return DataBatch([utterances],
                         [intents],
                         pad=0,
                         index=indices,
                         bucket_key=self.buckets[i],
                         provide_data=[DataDesc(name=self.data_name, shape=utterances.shape, layout=self.layout)],
                         provide_label=[DataDesc(name=self.label_name, shape=intents.shape, layout=self.layout)])


batch_size=12

train_iter = BucketUtteranceIter(X_train, Y_train, batch_size, buckets=[32,64,128,256,512,1024])
test_iter = BucketUtteranceIter(X_test, Y_test, batch_size, buckets=[32,64,128,256,512, 1024])

for i, batch in enumerate(train_iter):
    if i < 1:
        print("\nBatch {} Bucket size {}\nData\n {} \nLabel\n {}\n".format(i, batch.bucket_key, batch.data, batch.label))
train_iter.reset()

import mxnet as mx


def bucketed_module(train_iter, vocab_size, dropout, num_label, smooth_alpha, default_bucket_key, context):
    """
    :param train_iter:
    :param vocab_size:
    :param dropout:
    :param num_label:
    :param smooth_alpha:
    :param default_bucket_key:
    :param context:
    :return:
    """

    def sym_gen(seq_len):
        """
        :param seq_len: bucket size
        :return: symbol for neural network architecture
        """

        def conv(data, num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=None, suffix=''):
            conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad,
                                      no_bias=True, name='%s%s_conv2d' % (name, suffix))
            bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' % (name, suffix), fix_gamma=True)
            act = mx.sym.Activation(data=bn, act_type='relu', name='%s%s_relu' % (name, suffix))
            return act

        def conv_block(data, num_filter, name):
            conv1 = conv(data, kernel=(1, 3), num_filter=num_filter, pad=(0, 1), name='conv1' + str(name))
            conv2 = conv(conv1, kernel=(1, 3), num_filter=num_filter, pad=(0, 1), name='conv2' + str(name))
            return conv2

        X_shape = (train_iter.batch_size, seq_len)
        Y_shape = (train_iter.batch_size,)

        print("\nNetwork architecture for bucket size {}\n".format(seq_len))

        data = mx.sym.Variable(name="utterance")
        softmax_label = mx.sym.Variable(name="intent")
        print("data_input: ", data.infer_shape(utterance=X_shape)[1][0])
        print("label input: ", softmax_label.infer_shape(intent=Y_shape)[1][0])

        # Embed each character to 16 channels
        embedded_data = mx.sym.Embedding(data, input_dim=vocab_size, output_dim=16, name='embedding')
        embedded_data = mx.sym.Reshape(mx.sym.transpose(embedded_data, axes=(0, 2, 1)), shape=(0, 0, 1, -1), name='embedding_reshape')
        print("embed layer output shape: ", embedded_data.infer_shape(utterance=X_shape)[1][0])

        # Temporal Convolutional Layer (without activation)
        temp_conv = mx.sym.Convolution(embedded_data, kernel=(1, 3), num_filter=64, pad=(0, 1), name='temp_conv')
        print("Temp conv output shape: ", temp_conv.infer_shape(utterance=X_shape)[1][0])

        # Create convolutional blocks with pooling in-between
        block = conv_block(temp_conv, num_filter=64, name='block1_1')
        block = conv_block(block, num_filter=64, name='block1_2')
        pool = mx.sym.Pooling(block, kernel=(1, 3), stride=(1, 2), pad=(0, 1), pool_type='max')
        print("Block 1 output shape: {}".format(pool.infer_shape(utterance=X_shape)[1][0]))

        block = conv_block(pool, num_filter=128, name='block2_1')
        block = conv_block(block, num_filter=128, name='block2_2')
        pool = mx.sym.Pooling(block, kernel=(1, 3), stride=(1, 2), pad=(0, 1), pool_type='max')
        print("Block 2 output shape: {}".format(pool.infer_shape(utterance=X_shape)[1][0]))

        block = conv_block(pool, num_filter=256, name='block3_1')
        block = conv_block(block, num_filter=256, name='block3_2')
        pool = mx.sym.Pooling(block, kernel=(1, 3), stride=(1, 2), pad=(0, 1), pool_type='max')
        print("Block 3 output shape: {}".format(pool.infer_shape(utterance=X_shape)[1][0]))

        block = conv_block(pool, num_filter=512, name='block4_1')
        block = conv_block(block, num_filter=512, name='block4_2')
        print("Block 4 output shape: {}".format(block.infer_shape(utterance=X_shape)[1][0]))

        pool_k = seq_len // 8
        print("{0} pool kernel size {1}, stride 1".format('avg', pool_k))
        block = mx.sym.flatten(mx.sym.Pooling(block, kernel=(1, pool_k), stride=(1, 1), pad=(0, 0), pool_type='avg'), name='final_pooling')
        print("flattened pooling output shape: {}".format(block.infer_shape(utterance=X_shape)[1][0]))
        block = mx.sym.Dropout(block, p=dropout, name='dropout')
        print("dropout layer output shape: {}".format(block.infer_shape(utterance=X_shape)[1][0]))

        output = mx.sym.FullyConnected(block, num_hidden=num_label, flatten=True, name='output')
        sm = mx.sym.SoftmaxOutput(output, softmax_label, smooth_alpha, name='softmax_ce_loss')
        print("softmax output shape: {}".format(sm.infer_shape(utterance=X_shape)[1][0]))

        return sm, ('utterance',), ('intent',)

    return mx.mod.BucketingModule(sym_gen=sym_gen, default_bucket_key=default_bucket_key, context=context)



module = bucketed_module(train_iter,
                         vocab_size=150,
                         dropout=0.02,
                         num_label=4,
                         smooth_alpha=0.004,
                         default_bucket_key=train_iter.default_bucket_key,
                         context=mx.cpu())



# Reduce learning rate every 3 epochs
batches_per_epoch = int(sum([len(bucket) for bucket in train_iter.ndsent])/batch_size)
step = 3 * batches_per_epoch
schedule = mx.lr_scheduler.FactorScheduler(step=step, factor=0.97)

# Initialize convolutional filter weights using MSRAPRelu to aid training deeper architectures
init = mx.initializer.Mixed(patterns=['conv2d_weight', '.*'],
                            initializers=[mx.initializer.MSRAPrelu(factor_type='avg', slope=0.25),
                                          mx.initializer.Normal(sigma=0.02)])

# Learn network weights from data
module.fit(train_data=train_iter,
           eval_data=test_iter,
           eval_metric=mx.metric.Accuracy(),
           optimizer='sgd',
           optimizer_params={'learning_rate': 0.06,
                             'momentum': 0.93,
                             'lr_scheduler': schedule},
           # initializer=init,
           num_epoch=30)