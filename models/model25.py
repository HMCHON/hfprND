import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Concatenate, BatchNormalization, Activation, Dropout
from tensorflow.keras.models import Model


class model25:
    def __init__(self, input_shape1, n_classes):
        self.input_shape1 = input_shape1
        self.n_classes = n_classes
        self.model = self.build_model()

    def build_model(self):
        # 두 개의 서로 다른 입력 정의
        inputs1 = Input(shape=(self.input_shape1[0],34))
        inputs2 = Input(shape=(self.input_shape1[0], 6)) # left arm
        inputs3 = Input(shape=(self.input_shape1[0], 6)) # right arm
        inputs4 = Input(shape=(self.input_shape1[0],8)) # upper body
        inputs5 = Input(shape=(self.input_shape1[0],12))# lower body
        inputs6 = Input(shape=(self.input_shape1[0],10)) # head

        # 각 입력에 대해 별도의 컨볼루션 레이어
        whole_body_features = Conv1D(filters=128, kernel_size=self.input_shape1[0], activation=None, padding='valid')(inputs1)
        whole_body_features = BatchNormalization()(whole_body_features)
        whole_body_features = Activation('relu')(whole_body_features)

        left_arm_features = Conv1D(filters=128, kernel_size=self.input_shape1[0], activation=None, padding='valid')(inputs2)
        left_arm_features = BatchNormalization()(left_arm_features)
        left_arm_features = Activation('relu')(left_arm_features)

        right_arm_features = Conv1D(filters=128, kernel_size=self.input_shape1[0], activation=None, padding='valid')(inputs3)
        right_arm_features = BatchNormalization()(right_arm_features)
        right_arm_features = Activation('relu')(right_arm_features)

        upper_body_features = Conv1D(filters=128, kernel_size=self.input_shape1[0], activation=None, padding='valid')(inputs4)
        upper_body_features = BatchNormalization()(upper_body_features)
        upper_body_features = Activation('relu')(upper_body_features)

        lower_body_features = Conv1D(filters=128, kernel_size=self.input_shape1[0], activation=None, padding='valid')(inputs5)
        lower_body_features = BatchNormalization()(lower_body_features)
        lower_body_features = Activation('relu')(lower_body_features)

        head_features = Conv1D(filters=128, kernel_size=self.input_shape1[0], activation=None, padding='valid')(inputs6)
        head_features = BatchNormalization()(head_features)
        head_features = Activation('relu')(head_features)


        # 두 특징을 합치기
        output_list = [whole_body_features, left_arm_features, right_arm_features, upper_body_features, lower_body_features, head_features]
        concatenated_features = tf.concat(output_list, axis=1)

        # 2번째 1D-Convolution layer
        conv1D = Conv1D(filters=128, kernel_size=3, activation=None, padding='same')(concatenated_features)
        conv1D = BatchNormalization()(conv1D)
        conv1D = Activation('relu')(conv1D)

        # 3번째 1D-Convolution layer
        conv1D = Conv1D(filters=128, kernel_size=3, activation=None, padding='same')(conv1D)
        conv1D = BatchNormalization()(conv1D)
        conv1D = Activation('relu')(conv1D)

        # LSTM 레이어
        sequential_features = LSTM(128)(conv1D)

        # 분류 레이어
        dense = Dense(128, activation='relu')(sequential_features)
        dropout = Dropout(0.5)(dense)
        outputs = Dense(self.n_classes, activation='softmax')(dropout)

        # 모델 생성
        model = Model(inputs=[inputs1, inputs2, inputs3, inputs4, inputs5, inputs6], outputs=outputs)
        return model

    def get_model(self):
        return self.model