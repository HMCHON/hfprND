# model.py
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Concatenate
from tensorflow.keras.models import Model


class model2:
    def __init__(self, input_shape, n_classes):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.model = self.build_model()

    def build_model(self):
        inputs = Input(shape=self.input_shape)
        upper_body_features = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
        lower_body_features = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)

        # 특징 합치기
        concatenated_features = Concatenate()([upper_body_features, lower_body_features])
        concatenated_features = MaxPooling1D(pool_size=2)(concatenated_features)

        # LSTM 레이어
        sequential_features = LSTM(128)(concatenated_features)

        # 분류 레이어
        outputs = Dense(self.n_classes, activation='softmax')(sequential_features)

        # 모델 생성
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def get_model(self):
        return self.model
