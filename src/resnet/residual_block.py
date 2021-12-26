import tensorflow as tf


def basic_block(x, filter_num, stride=1, name=None):
    residual = x

    if stride != 1:
        residual = tf.keras.layers.Conv2D(filters=filter_num,
                                          kernel_size=(1, 1),
                                          strides=stride, name=f'{name}/conv_ds')(residual)
        residual = tf.keras.layers.BatchNormalization(
            name=f'{name}/bn_ds')(residual)

    x = tf.keras.layers.Conv2D(filters=filter_num,
                               kernel_size=(3, 3),
                               strides=stride,
                               padding='same',
                               name=f'{name}/conv1')(x)
    x = tf.keras.layers.BatchNormalization(name=f'{name}/bn1')(x)
    x = tf.keras.layers.ReLU(name=f'{name}/relu1')(x)
    x = tf.keras.layers.Conv2D(filters=filter_num,
                               kernel_size=(3, 3),
                               strides=1,
                               padding='same',
                               name=f'{name}/conv2')(x)
    x = tf.keras.layers.BatchNormalization(name=f'{name}/bn2')(x)
    x = tf.keras.layers.add([residual, x], name=f'{name}/add')
    x = tf.keras.layers.ReLU(name=f'{name}/relu2')(x)

    return x


def basic_block_layer(x, filter_num, blocks, stride=1, name=None):
    x = basic_block(x, filter_num, stride=stride, name=f'{name}/basic_block1')

    for i in range(1, blocks):
        x = basic_block(x, filter_num, stride=1,
                        name=f'{name}/basic_block{i + 1}')

    return x


def bottleneck_block(x, filter_num, stride=1, conv_shortcut=True, name=None):
    residual = x

    if conv_shortcut:
        residual = tf.keras.layers.Conv2D(filters=4 * filter_num,
                                          kernel_size=(1, 1),
                                          strides=stride, name=f'{name}/conv_ds')(residual)
        residual = tf.keras.layers.BatchNormalization(
            name=f'{name}/bn_ds')(residual)

    x = tf.keras.layers.Conv2D(filters=filter_num,
                               kernel_size=(1, 1),
                               strides=stride,
                               padding='same',
                               name=f'{name}/conv1')(x)
    x = tf.keras.layers.BatchNormalization(name=f'{name}/bn1')(x)
    x = tf.keras.layers.ReLU(name=f'{name}/relu1')(x)

    x = tf.keras.layers.Conv2D(filters=filter_num,
                               kernel_size=(3, 3),
                               strides=1,
                               padding='same',
                               name=f'{name}/conv2')(x)
    x = tf.keras.layers.BatchNormalization(name=f'{name}/bn2')(x)
    x = tf.keras.layers.ReLU(name=f'{name}/relu2')(x)

    x = tf.keras.layers.Conv2D(filters=4 * filter_num,
                               kernel_size=(1, 1),
                               strides=1,
                               padding='same',
                               name=f'{name}/conv3')(x)
    x = tf.keras.layers.BatchNormalization(name=f'{name}/bn3')(x)

    x = tf.keras.layers.add([residual, x], name=f'{name}/add')
    x = tf.keras.layers.ReLU(name=f'{name}/relu3')(x)

    return x


def bottleneck_block_layer(x, filter_num, blocks, stride=1, name=None):
    x = bottleneck_block(x, filter_num, stride=stride, name=f'{name}/bottleneck_block1')

    for i in range(1, blocks):
        x = bottleneck_block(x, filter_num, stride=1, conv_shortcut=False,
                             name=f'{name}/bottleneck_block{i + 1}')

    return x
