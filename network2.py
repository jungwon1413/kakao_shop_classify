# -*- coding: utf-8 -*-
# Copyright 2017 Kakao, Recommendation Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

import tensorflow as tf

import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers.merge import dot
from keras.layers import Add, Dense, Input, MaxPooling2D, AveragePooling2D, SeparableConv2D
from keras.layers.convolutional import Conv2D
from keras.layers.core import Reshape

from keras.layers.embeddings import Embedding
from keras.layers.core import Dropout, Activation, Flatten
from keras.layers import BatchNormalization as BN

from misc import get_logger, Option
opt = Option('./config.json')


def top1_acc(x, y):
    return keras.metrics.top_k_categorical_accuracy(x, y, k=1)

def Block(inp_x, num_channels, filter_size, activation, padding, dropout_rate=False, apply_maxpool=False):
    identity_x = inp_x
    x = inp_x    

    x = Conv2D(
            num_channels, kernel_size=1, activation=activation, padding=padding)(x)
    x = BN()(x)
    x = Activation(activation)(x)
    
    
    x = SeparableConv2D(
            num_channels//2, kernel_size=filter_size, activation=activation, padding=padding)(x)
    x = BN()(x)
    x = Activation(activation)(x)
    
    
    x = Conv2D(num_channels, kernel_size=filter_size, activation=activation, padding=padding)(x)
    x = BN()(x)

    # Shortcut value sum up
    x = Add()([x, identity_x])
    x = Activation(activation)(x)
    
    if apply_maxpool:
        x = MaxPooling2D(apply_maxpool)(x)
        
    if dropout_rate:
        x = Dropout(rate=dropout_rate)(x)
    
    return x


class TextOnly:
    def __init__(self):
        self.logger = get_logger('fasttext')
        
    def get_model(self, num_classes, activation='sigmoid'):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1
        filter_size = 3

        with tf.device('/gpu:1'):
            embd = Embedding(voca_size, opt.embd_size, name='uni_embd')

            t_uni = Input((max_len,), name="input_1")
            t_uni_embd = embd(t_uni)  # token
            print("t_uni_embd:", t_uni_embd.shape)

            w_uni = Input((max_len,), name="input_2")
            w_uni_mat = Reshape((max_len, 1))(w_uni)  # weight
            print("w_uni_mat:", w_uni_mat)

            uni_embd_mat = dot([t_uni_embd, w_uni_mat], axes=1)
            print("uni_embd_mat:", uni_embd_mat.shape)
            #uni_embd = Reshape((opt.embd_size,))(uni_embd_mat)
            uni_embd = Reshape((opt.embd_size//max_len, max_len//4, 4))(uni_embd_mat)
            print("uni_embd:", uni_embd.shape)

            #embd_out = Dropout(rate=0.5)(uni_embd)
            #print(uni_embd.shape)
            
            # Model w/ residual connection
            # Layer 1
            max_pool_filter = (2,2)
            x = Block(uni_embd, opt.embd_size//32, filter_size, 'relu', 'same', 0.5, max_pool_filter)
            x = Block(x, opt.embd_size//16, filter_size, 'relu', 'same', 0.5)
            x = Block(x, opt.embd_size//8, filter_size, 'relu', 'same', 0.5)
            """
            x = Conv2D(opt.embd_size//32, kernel_size=1, activation='relu', padding="same")(uni_embd)
            x = BN()(x)
            x = Activation('relu', name='relu1_1')(x)
            x = SeparableConv2D(opt.embd_size//16, kernel_size=filter_size, activation='relu', padding="same")(x)
            x = BN()(x)
            x = Activation('relu', name='relu1_2')(x)
            x = Conv2D(opt.embd_size//32, kernel_size=filter_size, activation='relu', padding="same")(x)
            x = BN()(x)
            x = Activation('relu', name='relu1_3')(x)
            x = Dropout(rate=0.5)(x)       # dropout after activation
            """

            
            # Layer 2
            max_pool_filter = (2,1)
            x = Block(x, opt.embd_size//16, filter_size, 'relu', 'same', 0.5, max_pool_filter)
            x = Block(x, opt.embd_size//8, filter_size, 'relu', 'same', 0.5)
            x = Block(x, opt.embd_size//4, filter_size, 'relu', 'same', 0.5)


            # Layer 3
            x = Block(x, opt.embd_size//8, filter_size, 'relu', 'same', 0.5)
            x = Block(x, opt.embd_size//4, filter_size, 'relu', 'same', 0.5)
            x = Block(x, opt.embd_size//2, filter_size, 'relu', 'same', 0.5)


            # Layer 4
            x = Block(x, opt.embd_size//4, filter_size, 'relu', 'same', 0.5)
            x = Block(x, opt.embd_size//2, filter_size, 'relu', 'same', 0.5)
            x = Block(x, opt.embd_size, filter_size, 'relu', 'same', 0.5)

            
            # Layer 5
            x = Block(x, opt.embd_size//2, filter_size, 'relu', 'same', 0.5)
            x = Block(x, opt.embd_size, filter_size, 'relu', 'same', 0.5)
            x = Block(x, opt.embd_size*2, filter_size, 'relu', 'same', 0.5)

            """
            # Layer 6 - Channel information restore(decoding)
            x = Conv2D(opt.embd_size, kernel_size=1, activation='relu', border_mode="same")(x)
            x = BN()(x)
            x = Activation('relu', name='relu_out_1')(x)
            x = Conv2D(opt.embd_size*2, kernel_size=filter_size, activation='relu', border_mode="same")(x)
            x = BN()(x)
            x = Activation('relu', name='relu_out_2')(x)
            x = Conv2D(opt.embd_size, kernel_size=filter_size, activation='relu', border_mode="same")(x)
            x = BN()(x)
            x = Activation('relu', name='relu_out_3')(x)
            """

            # Output Layer
            x = AveragePooling2D(pool_size=(4, 4))(x)
            x = Flatten()(x)
            #x = Dense(4096, activation='relu')(x)
            #x = Dropout(0.5)(x)
            #x = Dense(4096, activation='relu')(x)
            #x = Dropout(0.5)(x)
            output = Dense(num_classes, activation=activation)(x)
            
            model = Model(inputs=[t_uni, w_uni], outputs=output)
            optm = keras.optimizers.Nadam(opt.lr)
            model.compile(loss='binary_crossentropy',
                        optimizer=optm,
                        metrics=[top1_acc])
            model.summary(print_fn=lambda x: self.logger.info(x))
        return model
