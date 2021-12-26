from tensorflow.keras.layers import (Concatenate, Conv2DTranspose, LeakyReLU,
                                     UpSampling2D)


def conv_transpose_norm_block(x, filters, ker_reg, name, norm=None):
    x = Conv2DTranspose(filters, 3, padding='same',
                        name=f'{name}/conv_transpose', kernel_regularizer=ker_reg)(x)

    if norm is not None:
        x = norm(name=f'{name}/norm')(x)

    x = LeakyReLU(alpha=0.2, name=f'{name}/leaky_relu')(x)

    return x


def upscale_block_resnet(x1, x2, filters, name, ker_reg, norm=None):
    if x2 is not None:
        x = Concatenate(name=f'{name}/concatenate')([x1, x2])
    else:
        x = x1

    x = UpSampling2D(
        size=(2, 2), interpolation='bilinear', name=f'{name}/upsampling2d')(x)
    x = conv_transpose_norm_block(
        x, filters, ker_reg, f'{name}/block1', norm)
    x = conv_transpose_norm_block(
        x, filters, ker_reg, f'{name}/block2', norm)

    return x
