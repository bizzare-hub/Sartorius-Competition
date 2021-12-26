import tensorflow as tf

from .residual_block import basic_block_layer, bottleneck_block_layer


def ResNet(layer_params, block=basic_block_layer,
           input_shape=(256, 448, 3), name=None, num_classes=None):
    inputs = tf.keras.Input(input_shape)

    x = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=(7, 7),
                               strides=2,
                               padding='same',
                               name='init_conv')(inputs)
    x = tf.keras.layers.BatchNormalization(name='init_bn')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                  strides=2,
                                  padding='same',
                                  name='max_pool')(x)

    x = block(x, filter_num=64,
              blocks=layer_params[0], name='block1')
    x = block(x, filter_num=128,
              blocks=layer_params[1],
              stride=2, name='block2')
    x = block(x, filter_num=256,
              blocks=layer_params[2],
              stride=2, name='block3')
    x = block(x, filter_num=512,
              blocks=layer_params[3],
              stride=2, name='block4')

    if num_classes is not None:
        x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = tf.keras.layers.Dense(units=num_classes, name='dense')(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name=name)


def ResNet18(input_shape=(256, 448, 3), num_classes=None):
    return ResNet([2, 2, 2, 2],
                  input_shape=input_shape,
                  name='ResNet18',
                  num_classes=num_classes)


def ResNet34(input_shape=(256, 448, 3), num_classes=None):
    return ResNet([3, 4, 6, 3],
                  input_shape=input_shape,
                  name='ResNet34',
                  num_classes=num_classes)


def ResNet50(input_shape=(256, 448, 3), num_classes=None):
    return ResNet([3, 4, 6, 3],
                  bottleneck_block_layer,
                  input_shape=input_shape,
                  name='ResNet50',
                  num_classes=num_classes)


def ResNet101(input_shape=(256, 448, 3), num_classes=None):
    return ResNet([3, 4, 23, 3],
                  bottleneck_block_layer,
                  input_shape=input_shape,
                  name='ResNet101',
                  num_classes=num_classes)
