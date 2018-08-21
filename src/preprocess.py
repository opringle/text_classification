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

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Preprocess csvs for mxnet example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train-url', type=str, help='url to download csv file',
                    default='https://github.com/mhjabreel/CharCNN/raw/master/data/ag_news_csv/train.csv')
parser.add_argument('--test-url', type=str, help='url to download csv file',
                    default='https://github.com/mhjabreel/CharCNN/raw/master/data/ag_news_csv/test.csv')


def preprocess(df):
    """
    :param df: pandas dataframe
    :return: preprocessed pandas dataframe
    """
    index_to_label = {1: 'World', 2: 'Sports', 3: 'Business', 4: 'Sci/Tech'}
    df['label'] = df['indexed_label'].map(index_to_label)
    df.drop('indexed_label', inplace=True, axis=1)
    return df


def save(df, filename):
    """
    :param df: pandas dataframe
    :param filename: path to output file
    """
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    df.to_csv(filename, index=None)


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

    # Save to file
    save(train_df, '../data/ag_news/train.csv')
    save(test_df, '../data/ag_news/test.csv')
