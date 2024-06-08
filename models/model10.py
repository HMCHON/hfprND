import tensorflow as tf
from tensorflow.keras.layers import LSTM, Conv1DTranspose, Input, Conv1D, Dense, Concatenate, BatchNormalization, Activation, Dropout, Flatten
from tensorflow.keras.models import Model
import numpy as np

class model10:
    def __init__(self, input_shape1, n_classes):
        self.input_shape1 = input_shape1
        self.n_classes = n_classes
        self.model = self.build_model()

    def build_model(self):
        # 두 개의 서로 다른 입력 정의
        inputs1 = Input(shape=(self.input_shape1[0], 34))

        # 각 입력에 대해 별도의 컨볼루션 레이어 ( 1번쨰 1D-convolutional layer )
        whole_body_features = Conv1D(filters=128, kernel_size=self.input_shape1[0], activation=None, padding='valid')(inputs1)
        whole_body_features = BatchNormalization()(whole_body_features)
        whole_body_features = Activation('relu')(whole_body_features)

        # 2번쨰 1D-convolutional layer
        conv1D = Conv1D(filters=128, kernel_size=self.input_shape1[0], activation=None, padding='same')(whole_body_features)
        conv1D = BatchNormalization()(conv1D)
        conv1D = Activation('relu')(conv1D)

        # 3번쨰 1D-convolutional layer
        conv1D = Conv1D(filters=128, kernel_size=self.input_shape1[0], activation=None, padding='same')(conv1D)
        conv1D = BatchNormalization()(conv1D)
        conv1D = Activation('relu')(conv1D)

        # 4번쨰 1D-convolutional layer
        conv1D = Conv1D(filters=128, kernel_size=self.input_shape1[0], activation=None, padding='same')(conv1D)
        conv1D = BatchNormalization()(conv1D)
        conv1D = Activation('relu')(conv1D)

        # 5번쨰 1D-convolutional layer
        conv1D = Conv1D(filters=128, kernel_size=self.input_shape1[0], activation=None, padding='same')(conv1D)
        conv1D = BatchNormalization()(conv1D)
        conv1D = Activation('relu')(conv1D)

        # 6번쨰 1D-convolutional layer
        conv1D = Conv1D(filters=128, kernel_size=self.input_shape1[0], activation=None, padding='same')(conv1D)
        conv1D = BatchNormalization()(conv1D)
        conv1D = Activation('relu')(conv1D)

        # Flatten 레이어를 사용하여 1차원으로 변환
        flatten = Flatten()(conv1D)
        # sequential_features = LSTM(128)(conv1D)

        # 분류 레이어
        dense = Dense(128, activation='relu')(flatten)
        dropout = Dropout(0.5)(dense)
        outputs = Dense(self.n_classes, activation='softmax')(dropout)

        # 모델 생성
        model = Model(inputs=[inputs1], outputs=outputs)
        return model

    def get_model(self):
        return self.model
