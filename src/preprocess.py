#!/usr/bin/env python

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# -*- coding: utf-8 -*-

import pandas as pd
import logging
import argparse
import os
from collections import Counter
import itertools
import pickle


logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Preprocess csvs for mxnet example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train-url', type=str, help='url to download csv file',
                    default='https://github.com/mhjabreel/CharCNN/raw/master/data/ag_news_csv/train.csv')
parser.add_argument('--test-url', type=str, help='url to download csv file',
                    default='https://github.com/mhjabreel/CharCNN/raw/master/data/ag_news_csv/test.csv')


class CharacterPreprocessor:
    def __init__(self, char_to_index={}, index_to_char=[], intent_to_index={}, index_to_intent=[], unseen_character=-1,
                 unseen_label=-1):
        """
        :param char_to_index: dictionary where key is character and value is indexed character
        :param index_to_char: list where position is index and value is char
        :param intent_to_index: dictionary where key is intent and value is indexed intent
        :param index_to_intent: list where position is index and value is intent
        :param unseen_character: value to use when indexing a character which is not in char_to_index
        :param unseen_label: value to use when indexing an intent which is not in intent_to_index
        """
        self.char_to_index = char_to_index
        self.index_to_char = index_to_char
        self.intent_to_index = intent_to_index
        self.index_to_intent = index_to_intent
        self.unseen_character = unseen_character
        self.unseen_label = unseen_label

    @staticmethod
    def split_utterance(utterance):
        """
        :param utterance: string
        :return: list of string, split into characters
        """
        return list(utterance.lower())

    @staticmethod
    def build_vocab(data, max_vocab_size, depth=1):
        """
        :param data: list of data
        :param depth: how many levels of nesting there are in list
        :param max_vocab_size: limit vocabulary to n most popular occurrences
        :return: dict and list mapping data to indices
        """
        if depth > 1:
            data = list(itertools.chain.from_iterable(data)) # Make list 1D
        data_counts = Counter(data)  # Count occurrences of each word in the list

        vocabulary_inv = [x[0] for x in data_counts.most_common(max_vocab_size)]
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
        return vocabulary, vocabulary_inv

    def fit(self, utterances, labels, max_vocab_size):
        """
        Build vocabulary from data
        :param utterances: list of raw utterances
        :param max_vocab_size: max number of characters in the vocabulary
        :param labels: list of raw labels
        """
        split_utterances = [self.split_utterance(utterance) for utterance in utterances]
        self.char_to_index, self.index_to_char = self.build_vocab(split_utterances, max_vocab_size, depth=2)
        self.intent_to_index, self.index_to_intent = self.build_vocab(labels, max_vocab_size, depth=1)

    def transform_utterance(self, utterance):
        """
        :param utterance: raw utterance string
        :return: preprocessed utterance
        """
        split_utterance = self.split_utterance(utterance)
        indexed_utterance = [self.char_to_index.get(token, self.unseen_character) for token in split_utterance]
        return indexed_utterance

    def transform_label(self, label):
        """
        :param label: raw intent label
        :return: indexed intent label
        """
        return self.intent_to_index.get(label, self.unseen_label)


def preprocess(df):
    """
    :param df: pandas dataframe
    :return: preprocessed pandas dataframe
    """
    index_to_label = {1: 'World', 2: 'Sports', 3: 'Business', 4: 'Sci/Tech'}
    df['label'] = df['indexed_label'].map(index_to_label)
    df.drop('indexed_label', inplace=True, axis=1)
    return df


def index_features(train_df, test_df, preprocessor):
    """
    :param train_df: training data in pandas df
    :param test_df: test data in pandas df
    :return: indexed feature dataframes
    """
    # Fit prepreprocessor to training data
    preprocessor.fit(train_df.description.tolist(), train_df.label.tolist(), max_vocab_size=200)

    # Transform utterances
    train_df['X'] = train_df['description'].apply(lambda x: preprocessor.transform_utterance(x))
    train_df['Y'] = train_df['label'].apply(lambda x: preprocessor.transform_label(x))

    test_df['X'] = test_df['description'].apply(lambda x: preprocessor.transform_utterance(x))
    test_df['Y'] = test_df['label'].apply(lambda x: preprocessor.transform_label(x))

    return train_df, test_df


def save(df, filename):
    """
    :param df: pandas dataframe
    :param filename: path to output file
    """
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    df.to_csv(filename, index=None)


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # Parse args
    args = parser.parse_args()

    # Download data
    train_df = pd.read_csv(args.train_url,
                           names=['indexed_label', 'title', 'description'])
    test_df = pd.read_csv(args.test_url,
                           names=['indexed_label', 'title', 'description'])

    # Preprocess
    train_df = preprocess(train_df)
    test_df = preprocess(test_df)

    preprocessor = CharacterPreprocessor()
    train_df, test_df = index_features(train_df, test_df, preprocessor)

    # Save to file
    save_obj(preprocessor.char_to_index, '../data/ag_news_char/char_to_index')
    save_obj(preprocessor.intent_to_index, '../data/ag_news_char/intent_to_index')
    train_df.to_pickle('../data/ag_news_char/train.pickle')
    test_df.to_pickle('../data/ag_news_char/test.pickle')
