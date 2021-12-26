from tensorflow.keras.layers import (Conv2D, MaxPool2D, Add,
                                     ReLU, UpSampling2D)


def residual_conv_unit(x, filter_num, ker_reg,
                       conv_shortcut=False, name=None):
    residual = x

    if conv_shortcut:
        residual = Conv2D(filters=filter_num,
                          kernel_size=(1, 1),
                          strides=1,
                          padding='same',
                          name=f'{name}/conv_shortcut',
                          kernel_regularizer=ker_reg)(residual)
        residual = ReLU(name=f'{name}/relu_shortcut')(residual)

    # x = ReLU(name=f'{name}/relu1')(x)
    # x = Conv2D(filters=filter_num,
    #            kernel_size=(3, 3),
    #            strides=1,
    #            padding='same',
    #            name=f'{name}/conv1',
    #            kernel_regularizer=ker_reg)(x)
    # x = ReLU(name=f'{name}/relu2')(x)
    # x = Conv2D(filters=filter_num,
    #            kernel_size=(3, 3),
    #            strides=1,
    #            padding='same',
    #            name=f'{name}/conv2',
    #            kernel_regularizer=ker_reg)(x)
    x = ReLU(name=f'{name}/relu1')(x)
    x = Conv2D(filters=filter_num,
               kernel_size=(1, 1),
               strides=1,
               padding='same',
               name=f'{name}/conv1',
               kernel_regularizer=ker_reg)(x)
    x = ReLU(name=f'{name}/relu2')(x)
    x = Conv2D(filters=filter_num,
               kernel_size=(3, 3),
               strides=1,
               padding='same',
               name=f'{name}/conv2',
               kernel_regularizer=ker_reg)(x)
    x = ReLU(name=f'{name}/relu3')(x)
    x = Conv2D(filters=filter_num,
               kernel_size=(1, 1),
               strides=1,
               padding='same',
               name=f'{name}/conv3',
               kernel_regularizer=ker_reg)(x)

    x = Add(name=f'{name}/add')([residual, x])

    return x


def chained_residual_pooling(x, filter_num, ker_reg, name=None):
    def residual_pooling(x, index):
        residual = x

        x = ReLU(name=f'{name}/relu{index}')(x)
        x = MaxPool2D(pool_size=(5, 5),
                      strides=1,
                      padding='same',
                      name=f'{name}/maxpool{index}')(x)
        # x = Conv2D(filters=filter_num,
        #            kernel_size=(3, 3),
        #            strides=1,
        #            padding='same',
        #            name=f'{name}/conv{index}',
        #            kernel_regularizer=ker_reg)(x)
        x = Conv2D(filters=filter_num,
                   kernel_size=(1, 1),
                   strides=1,
                   padding='same',
                   name=f'{name}/conv{index}',
                   kernel_regularizer=ker_reg)(x)
        x = Add(name=f'{name}/add{index}')([residual, x])

        return x

    x = ReLU(name=f'{name}/relu1')(x)

    for i in range(2, 5):
        x = residual_pooling(x, i)

    return x


def multi_resolution_fusion(x1, x2, filter_num, ker_reg, name=None):
    # x1 = Conv2D(filters=filter_num,
    #             kernel_size=(3, 3),
    #             strides=1,
    #             padding='same',
    #             name=f'{name}/low_conv',
    #             kernel_regularizer=ker_reg)(x1)
    x1 = Conv2D(filters=filter_num,
                kernel_size=(1, 1),
                strides=1,
                padding='same',
                name=f'{name}/low_conv',
                kernel_regularizer=ker_reg)(x1)
    x1 = UpSampling2D(
        size=(2, 2), interpolation='bilinear', name=f'{name}/upsampling2d')(x1)

    # x2 = Conv2D(filters=filter_num,
    #             kernel_size=(3, 3),
    #             strides=1,
    #             padding='same',
    #             name=f'{name}/high_conv',
    #             kernel_regularizer=ker_reg)(x2)
    x2 = Conv2D(filters=filter_num,
                kernel_size=(1, 1),
                strides=1,
                padding='same',
                name=f'{name}/high_conv',
                kernel_regularizer=ker_reg)(x2)
    x = Add(name=f'{name}/add')([x1, x2])

    return x


def refine_block(x1, x2, filters, ker_reg, conv_shortcut=False, name=None):
    x1 = residual_conv_unit(x1, filters, ker_reg, conv_shortcut,
                            name=f'{name}/low_residual_conv_unit1')
    x1 = residual_conv_unit(x1, filters, ker_reg,
                            name=f'{name}/low_residual_conv_unit2')

    if x2 is None:
        x = chained_residual_pooling(x1, filters, ker_reg,
                                     name=f'{name}/chained_residual_pooling')
        x = residual_conv_unit(x, filters, ker_reg,
                               name=f'{name}/residual_conv_unit3')

        return x
    else:
        x2 = residual_conv_unit(x2, filters, ker_reg, True,
                                name=f'{name}/high_residual_conv_unit1')
        x2 = residual_conv_unit(x2, filters, ker_reg,
                                name=f'{name}/high_residual_conv_unit2')

        x = multi_resolution_fusion(x1, x2, filters, ker_reg,
                                    name=f'{name}/multi_resolution_fusion')
        x = chained_residual_pooling(x, filters, ker_reg,
                                     name=f'{name}/chained_residual_pooling')
        x = residual_conv_unit(x, filters, ker_reg,
                               name=f'{name}/residual_conv_unit')

        return x
