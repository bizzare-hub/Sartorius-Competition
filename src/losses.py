import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.python.keras.losses import LossFunctionWrapper


def dice_loss(y_true, y_pred, smooth=1e-3, from_logits=True):
    if from_logits:
        y_pred = tf.nn.sigmoid(y_pred)

    y_true = tf.cast(y_true, tf.float32)

    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true * y_pred)
    true_area = tf.reduce_sum(y_true)
    pred_area = tf.reduce_sum(y_pred)

    return 1. - tf.divide(
        2. * intersection + smooth, true_area + pred_area + smooth)


def tversky_loss(y_true, y_pred,
                 alpha=0.7, smooth=1e-3, from_logits=True):
    if from_logits:
        y_pred = tf.nn.sigmoid(y_pred)

    y_true = tf.cast(y_true, tf.float32)

    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    true_pos = tf.reduce_sum(y_true * y_pred)
    false_neg = tf.reduce_sum(y_true * (1. - y_pred))
    false_pos = tf.reduce_sum((1. - y_true) * y_pred)

    return 1. - tf.divide(
        true_pos + smooth, true_pos + alpha * false_neg + (1. - alpha) * false_pos + smooth)


def binary_segmentation_loss(y_true, y_pred, lambd=0.5, from_logits=True):
    """CE + Tversky (or Dice)"""
    bce = tf.keras.losses.binary_crossentropy(
        y_true, y_pred, from_logits=from_logits)
    dice = dice_loss(y_true, y_pred, from_logits=from_logits)

    return tf.reduce_mean(bce) * lambd + (1. - lambd) * dice


def joint_segmentation_loss(y_true, y_pred, lambdas=(0.5, 0.5), from_logits=True):
    """Segmentation Loss for Multi-Channel classification"""
    y_true = tf.transpose(y_true, [3, 0, 1, 2])
    y_pred = tf.transpose(y_pred, [3, 0, 1, 2])
    lambdas = tf.reshape(lambdas, [-1, 1])

    def wrapper(inputs):
        y_true, y_pred, lambd = inputs

        loss = binary_segmentation_loss(
            y_true, y_pred, lambd, from_logits)

        return loss

    inputs = (y_true, y_pred, lambdas)

    losses = tf.map_fn(
        wrapper, inputs, fn_output_signature=tf.float32)

    return tf.reduce_mean(losses)


def _build_weights_mask(y_true, weights):
    arange = tf.range(0, limit=tf.shape(weights)[0])

    def wrapper(inputs):
        index, weight = inputs

        return tf.cast(y_true == index, tf.float32) * weight

    weights_mask = tf.map_fn(
        wrapper,
        (arange, weights),
        fn_output_signature=tf.float32
    )

    return tf.reduce_sum(weights_mask, axis=0)


def weighted_scce_loss(y_true, y_pred, weights):
    y_true = tf.cast(y_true, tf.int32)

    weights_mask = _build_weights_mask(y_true, weights)

    scce = SparseCategoricalCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE
    )

    loss = scce(y_true, y_pred) * tf.squeeze(weights_mask)

    return tf.reduce_mean(loss)


def categorical_segmentation_loss(y_true, y_pred,
                                  weights, alpha, lambd=0.5):
    """Combines weighted_scce_loss + tversky per-class loss"""
    y_true = tf.cast(y_true, tf.int32)

    weighted_scce = weighted_scce_loss(y_true, y_pred, weights)

    y_pred = tf.nn.softmax(y_pred, axis=-1)
    arange = tf.range(0, limit=tf.shape(weights)[0])

    def wrapper(index):
        y_true_class = tf.cast(y_true == index, tf.float32)
        y_pred_class = y_pred[..., index]

        return tversky_loss(
            y_true_class, y_pred_class, alpha=alpha, from_logits=False)

    tversky = tf.map_fn(
        wrapper,
        arange,
        fn_output_signature=tf.float32
    )

    return lambd * weighted_scce + (1. - lambd) * tversky


class JointSegmentationLoss(LossFunctionWrapper):
    def __init__(
        self,
        lambdas=(0.5, 0.5),
        from_logits=True
    ):
        super().__init__(
            joint_segmentation_loss,
            reduction=tf.keras.losses.Reduction.NONE,
            name='JointSegmentationLoss',
            lambdas=lambdas,
            from_logits=from_logits
        )


class JointTypeSegmentationLoss(LossFunctionWrapper):
    def __init__(
        self,
        lambdas=(0.5, 0.5),
        weights=[1., 1., 1.],
        from_logits=True
    ):
        weights = tf.constant(weights, dtype=tf.float32)

        segmentation_loss = JointSegmentationLoss(lambdas, from_logits)
        type_loss = SparseCategoricalCrossentropy(from_logits)  # reduction ?

        def loss_fn(y_true, y_pred):
            logits_type, logits = y_pred

            return weights * type_loss(y_true, logits_type) +\
                   segmentation_loss(y_true, logits)

        super().__init__(
            loss_fn,
            reduction=tf.keras.losses.Reduction.NONE,
            name='JointTypeSegmentationLoss'
        )


class WeightedSparseCCELoss(LossFunctionWrapper):
    def __init__(self, weights=[1., 1., 1.]):
        weights = tf.constant(weights, dtype=tf.float32)

        super().__init__(
            weighted_scce_loss,
            weights=weights,
            reduction=tf.keras.losses.Reduction.NONE,
            name='WeightedSparseCCELoss'
        )


class CategoricalSegmentationLoss(LossFunctionWrapper):
    def __init__(
        self,
        weights=[1., 1., 1.],
        alpha=0.7,
        lambd=0.5
    ):
        weights = tf.constant(weights, dtype=tf.float32)

        super().__init__(
            categorical_segmentation_loss,
            weights=weights,
            alpha=alpha,
            lambd=lambd,
            reduction=tf.keras.losses.Reduction.NONE,
            name='CategoricalSegmentationLoss'
        )
