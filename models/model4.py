import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Concatenate, BatchNormalization, Activation, Dropout
from tensorflow.keras.models import Model


class model4:
    def __init__(self, input_shape1, n_classes):
        self.input_shape1 = input_shape1
        self.n_classes = n_classes
        self.model = self.build_model()

    def build_model(self):
        # 두 개의 서로 다른 입력 정의
        inputs1 = Input(shape=(self.input_shape1[0],34))

        # 각 입력에 대해 별도의 컨볼루션 레이어
        whole_body_features = Conv1D(filters=64, kernel_size=self.input_shape1[0], activation=None, padding='valid')(inputs1)
        whole_body_features = BatchNormalization()(whole_body_features)
        whole_body_features = Activation('relu')(whole_body_features)

        # 생성된 (self.input_shape2[0], 64)의 특징맵에서 전역적 종속특징맵 생성
        conv1D = Conv1D(filters=64, kernel_size=3, activation=None, padding='same')(whole_body_features)
        conv1D = BatchNormalization()(conv1D)
        conv1D = Activation('relu')(conv1D)

        # LSTM 레이어
        sequential_features = LSTM(128)(conv1D)

        # 분류 레이어
        dense = Dense(128, activation='relu')(sequential_features)
        dropout = Dropout(0.5)(dense)
        outputs = Dense(self.n_classes, activation='softmax')(dropout)

        # 모델 생성
        model = Model(inputs=[inputs1], outputs=outputs)
        return model

    def get_model(self):
        return self.model