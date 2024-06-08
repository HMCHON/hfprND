# PoseModel.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, MaxPooling1D, LSTM, Dense, Dropout


class model1:
    def __init__(self, input_shape, n_classes):
        self.n_frames = input_shape[0]
        self.n_joints = input_shape[1]
        self.n_classes = n_classes
        self.model = self.build_model()

    def build_model(self):
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, padding='same', input_shape=(self.n_frames, self.n_joints)),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling1D(pool_size=2),

            Conv1D(filters=128, kernel_size=3, padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling1D(pool_size=2),

            LSTM(128, return_sequences=True),
            LSTM(128),

            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(self.n_classes, activation='softmax')
        ])
        return model

    def compile_model(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def get_model(self):
        return self.model

    def summary(self):
        return self.model.summary()
