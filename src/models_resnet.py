import tensorflow as tf
from tensorflow.keras.layers import (BatchNormalization, Conv2DTranspose,
                                     LayerNormalization, UpSampling2D,
                                     Conv2D)
from tensorflow.keras.regularizers import l2

from .custom_blocks import upscale_block_resnet
from src.refinenet.layers import refine_block
from src.resnet.models import ResNet18, ResNet34, ResNet50, ResNet101


class Identity(tf.keras.layers.Layer):
    def call(self, x):
        return x


def decoder_net(features, decode_filters, norm, reg_value, num_out_filters, name):
    norms = {
        'layer_norm': LayerNormalization,
        'batch_norm': BatchNormalization
    }

    ker_reg = l2(reg_value) if reg_value != 0. else None
    norm_cls = norms[norm]

    x, feat3, feat2, feat1 = features

    x = Conv2DTranspose(decode_filters, 1, padding='same',
                        name=f'{name}/conv_init', kernel_regularizer=ker_reg)(x)
    up0 = upscale_block_resnet(x, None, filters=decode_filters // 2,
                               name=f'{name}/up0', ker_reg=ker_reg, norm=norm_cls)
    up1 = upscale_block_resnet(up0, feat3, filters=decode_filters // 4,
                               name=f'{name}/up1', ker_reg=ker_reg, norm=norm_cls)
    up2 = upscale_block_resnet(up1, feat2, filters=decode_filters // 8,
                               name=f'{name}/up2', ker_reg=ker_reg, norm=norm_cls)
    up3 = upscale_block_resnet(up2, feat1, filters=decode_filters // 16,
                               name=f'{name}/up3', ker_reg=ker_reg, norm=norm_cls)

    x = Conv2DTranspose(num_out_filters, 3, padding='same',
                        name=f'{name}/conv_pred', kernel_regularizer=ker_reg)(up3)
    x = UpSampling2D(size=(2, 2), interpolation='bilinear',
                     name=f'{name}/upsample_pred')(x)

    return x


def decoder_refinenet(features, reg_value, num_out_filters, name=None):
    ker_reg = l2(reg_value) if reg_value != 0. else None

    x, feat3, feat2, feat1 = features

    up0 = refine_block(x, None, filters=512, conv_shortcut=True,
                       name=f'{name}/up0', ker_reg=ker_reg)
    up1 = refine_block(up0, feat3, filters=256, conv_shortcut=True,
                       name=f'{name}/up1', ker_reg=ker_reg)
    up2 = refine_block(up1, feat2, filters=256,
                       name=f'{name}/up2', ker_reg=ker_reg)
    up3 = refine_block(up2, feat1, filters=256,
                       name=f'{name}/up3', ker_reg=ker_reg)

    x = tf.keras.layers.Conv2DTranspose(num_out_filters, 3, padding='same',
                                        name=f'{name}/conv_pred', kernel_regularizer=ker_reg)(up3)
    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear',
                                     name=f'{name}/upsample_pred')(x)

    return x


def ResUNet18(input_shape=(256, 448, 3), decode_filters=512,
              norm='batch_norm', reg_value=0., num_out_filters=1):
    encoder = ResNet18(input_shape)

    names = [f'block{i}/basic_block2/add' for i in range(4, 0, -1)]
    x = [encoder.get_layer(n).output for n in names]
    x = decoder_net(x, decode_filters, norm,
                    reg_value, num_out_filters, name='Decoder')

    return tf.keras.Model(inputs=encoder.inputs, outputs=x)


def ResUNet34(input_shape=(256, 448, 3), decode_filters=512,
              norm='batch_norm', reg_value=0., num_out_filters=1):
    encoder = ResNet34(input_shape)

    block_n = [3, 4, 6, 3]
    names = [f'block{i}/basic_block{block_n[i - 1]}/add' for i in range(4, 0, -1)]
    x = [encoder.get_layer(n).output for n in names]
    x = decoder_net(x, decode_filters, norm,
                    reg_value, num_out_filters, name='Decoder')

    return tf.keras.Model(inputs=encoder.inputs, outputs=x)


def ResUNet50(input_shape=(256, 448, 3), decode_filters=512,
              norm='batch_norm', reg_value=0., num_out_filters=1):
    encoder = ResNet50(input_shape)

    block_n = [3, 4, 6, 3]
    names = [f'block{i}/bottleneck_block{block_n[i - 1]}/add' for i in range(4, 0, -1)]
    x = [encoder.get_layer(n).output for n in names]
    x = decoder_net(x, decode_filters, norm,
                    reg_value, num_out_filters, name='Decoder')

    return tf.keras.Model(inputs=encoder.inputs, outputs=x)


def ResUNet101(input_shape=(256, 448, 3), decode_filters=2048,
               norm='batch_norm', reg_value=0., num_out_filters=1):
    encoder = ResNet101(input_shape)

    block_n = [3, 4, 23, 3]
    names = [f'block{i}/bottleneck_block{block_n[i - 1]}/add' for i in range(4, 0, -1)]
    x = [encoder.get_layer(n).output for n in names]
    x = decoder_net(x, decode_filters, norm,
                    reg_value, num_out_filters, name='Decoder')

    return tf.keras.Model(inputs=encoder.inputs, outputs=x)


def RefineNet50(input_shape=(256, 448, 3), reg_value=0., num_out_filters=1):
    encoder = ResNet50(input_shape)

    block_n = [3, 4, 6, 3]
    names = [f'block{i}/bottleneck_block{block_n[i - 1]}/add' for i in range(4, 0, -1)]
    x = [encoder.get_layer(n).output for n in names]
    x = decoder_refinenet(x, reg_value, num_out_filters,
                          name='Decoder')

    return tf.keras.Model(inputs=encoder.inputs, outputs=x)


def RefineNet101(input_shape=(256, 448, 3), reg_value=0., num_out_filters=1):
    encoder = ResNet101(input_shape)

    block_n = [3, 4, 23, 3]
    names = [f'block{i}/bottleneck_block{block_n[i - 1]}/add' for i in range(4, 0, -1)]
    x = [encoder.get_layer(n).output for n in names]
    x = decoder_refinenet(x, reg_value, num_out_filters,
                          name='Decoder')

    return tf.keras.Model(inputs=encoder.inputs, outputs=x)


def ResUNet50Ext(
    input_shape=(256, 448, 3),
    decode_filters=512,
    norm='batch_norm',
    reg_value=0.,
    num_out_filters_embs=8,
    num_out_filters=1
):
    model = ResUNet50(
        input_shape, decode_filters, norm, reg_value, num_out_filters_embs)

    x = Conv2D(filters=4, kernel_size=4, padding='same')(model.output)
    x = Conv2D(filters=num_out_filters, kernel_size=1, padding='same')(x)

    embs = Identity(name='Embeddings')(model.output)
    logits = Identity(name='Logits')(x)

    output = tf.keras.layers.Concatenate(axis=-1, name='concat_to_output')([embs, logits])

    return tf.keras.Model(inputs=model.inputs, outputs=output)
