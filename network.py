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
from keras.layers import Add, Dense, Input, MaxPooling2D, GlobalMaxPooling2D, AveragePooling2D, SeparableConv2D
from keras.layers.convolutional import Conv2D
from keras.layers.core import Reshape

from keras.layers.embeddings import Embedding
from keras.layers.core import Dropout, Activation, Flatten
from keras.layers import BatchNormalization as BN

from misc import get_logger, Option
opt = Option('./config.json')


def top1_acc(x, y):
    return keras.metrics.top_k_categorical_accuracy(x, y, k=1)

def top5_acc(x, y):
    return keras.metrics.top_k_categorical_accuracy(x, y, k=5)

def Block(inp_x, num_channels, filter_size, activation, padding, dropout_rate=False, maxpool=False):
    x_1 = inp_x
    x_2 = inp_x
    x_skip = inp_x

    #### MAIN PATH 1 ####
    # First component of main path #1
    x_1 = Conv2D(num_channels, kernel_size=1, padding=padding)(x_1)
    x_1 = BN()(x_1)
    x_1 = Activation(activation)(x_1)
    
    # Second component of main path #1
    #x = SeparableConv2D(num_channels//2, kernel_size=filter_size, padding=padding)(x)
    x_1 = SeparableConv2D(num_channels//2, kernel_size=filter_size, padding=padding)(x_1)
    x_1 = BN()(x_1)
    x_1 = Activation(activation)(x_1)
    
    # Third component of main path #1
    x_1 = Conv2D(num_channels//2, kernel_size=1, padding=padding)(x_1)
    x_1 = BN()(x_1)


    #### MAIN PATH 2 ####
    # First component of main path #2
    x_2 = Conv2D(num_channels, kernel_size=1, padding=padding)(x_2)
    x_2 = BN()(x_2)
    x_2 = Activation(activation)(x_2)

    # Second component of main path #2
    x_2 = SeparableConv2D(num_channels//2, kernel_size=filter_size, padding=padding)(x_2)
    x_2 = BN()(x_2)
    x_2 = Activation(activation)(x_2)

    # Third component of main path #2
    x_2 = SeparableConv2D(num_channels//2, kernel_size=filter_size, padding=padding)(x_2)
    x_2 = BN()(x_2)
    x_2 = Activation(activation)(x_2)

    # Fourth component of main path #2
    x_2 = Conv2D(num_channels//2, kernel_size=1, padding=padding)(x_2)
    x_2 = BN()(x_2)


    #### SKIP CONNECTION ####
    x_skip = Conv2D(num_channels//2, kernel_size=1, padding=padding)(x_skip)
    x_skip = BN()(x_skip)

    if maxpool:
        x_1 = MaxPooling2D(maxpool)(x_1)
        x_2 = MaxPooling2D(maxpool)(x_2)
        x_skip = MaxPooling2D(maxpool)(x_skip)
    
    # Shortcut value sum up
    x = Add()([x_1, x_2, x_skip])
    x = Activation(activation)(x)
    
    if dropout_rate:
        x = Dropout(rate=dropout_rate)(x)
    
    return x


class TextOnly:
    def __init__(self):
<<<<<<< HEAD
        self.logger = get_logger('fasttext')
        
=======
        self.logger = get_logger('textonly')

>>>>>>> e6961dbdcb0dccb7d937393876a09fe3d44df26b
    def get_model(self, num_classes, activation='sigmoid'):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1
        filter_size = 3

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
        uni_embd = Reshape((opt.embd_size//max_len, max_len, 1))(uni_embd_mat)
        print("uni_embd:", uni_embd.shape)

            #embd_out = Dropout(rate=0.5)(uni_embd)
            #print(uni_embd.shape)
            
            # Model w/ residual connection
            # Layer 1
        mp_row = (2,1)
        mp_col = (1,2)
        x = Block(uni_embd, opt.embd_size//32, filter_size, 'relu', 'same', maxpool=mp_col)
        x = Block(x, opt.embd_size//16, filter_size, 'relu', 'same')
        x = Block(x, opt.embd_size//8, filter_size, 'relu', 'same')
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
        x = Block(x, opt.embd_size//16, filter_size, 'relu', 'same', maxpool=mp_row)
        x = Block(x, opt.embd_size//8, filter_size, 'relu', 'same')
        x = Block(x, opt.embd_size//4, filter_size, 'relu', 'same')


            # Layer 3
        x = Block(x, opt.embd_size//8, filter_size, 'relu', 'same', maxpool=mp_col)
        x = Block(x, opt.embd_size//4, filter_size, 'relu', 'same')
        x = Block(x, opt.embd_size//2, filter_size, 'relu', 'same')


            # Layer 4
        x = Block(x, opt.embd_size//4, filter_size, 'relu', 'same', maxpool=mp_row)
        x = Block(x, opt.embd_size//2, filter_size, 'relu', 'same')
        x = Block(x, opt.embd_size, filter_size, 'relu', 'same')

            
            # Layer 5
        x = Block(x, opt.embd_size//2, filter_size, 'relu', 'same', maxpool=mp_col)
        x = Block(x, opt.embd_size, filter_size, 'relu', 'same')
        x = Block(x, opt.embd_size*2, filter_size, 'relu', 'same')

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
        x = MaxPooling2D((4,4))(x)
        x = Flatten()(x)
        #x = Dense(4096, activation='relu')(x)
        #x = Dropout(0.5)(x)
        #x = Dense(4096, activation='relu')(x)
        #x = Dropout(0.5)(x)
        output = Dense(num_classes, activation=activation)(x)
            
        model = Model(inputs=[t_uni, w_uni], outputs=output)
        optm = keras.optimizers.Adam(lr=opt.lr, decay=opt.lr_decay_rate)
        model.compile(loss='categorical_crossentropy',
                        optimizer=optm,
                        metrics=[top1_acc, top5_acc])
        model.summary(print_fn=lambda x: self.logger.info(x))
        
        return model
