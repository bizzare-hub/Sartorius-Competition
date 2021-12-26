import tensorflow as tf
from tensorflow.keras.applications import ResNet50

from .models_resnet import decoder_net


def ResNetTF50(input_shape=(256, 448, 3)):
    x = inputs = tf.keras.Input(shape=input_shape)

    if input_shape[-1] != 3:
        x = tf.keras.layers.Conv2D(3, 1, strides=1, padding='same',
                                   use_bias=False, name='gate_conv')(x)

    x = ResNet50(input_shape=input_shape[:2] + (3,), weights='imagenet',
                 include_top=False)(x)


    return tf.keras.Model(inputs=inputs, outputs=x)


def ResUNetTF50(input_shape=(256, 448, 3), decode_filters=512,
                norm='batch_norm', reg_value=0., num_out_filters=1):
    encoder = ResNetTF50(input_shape)

    block_n = [3, 4, 6, 3]
    names = [f'conv{i + 1}_block{block_n[i - 1]}_out' for i in range(4, 0, -1)]
    x = [encoder.layers[-1].get_layer(n).output for n in names]
    x = decoder_net(x, decode_filters, norm,
                    reg_value, num_out_filters, name='Decoder')

    return tf.keras.Model(inputs=encoder.inputs, outputs=x)
