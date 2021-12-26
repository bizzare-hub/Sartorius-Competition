import tensorflow as tf
from tensorflow_addons.metrics import MeanMetricWrapper
from tensorflow.keras.metrics import AUC, Precision, Recall


# def _preprocess_input(y_pred):
#     mask = y_pred[..., 8:]
#
#     return mask


def average_precision_no_batch(inputs):
    thresholds = tf.range(
        start=0.5, limit=1., delta=0.05, dtype=tf.float32)
    n_thr = tf.cast(tf.shape(thresholds)[0], tf.float32)

    label, pred = inputs

    label_ravel = tf.reshape(label, [-1])
    pred_ravel = tf.reshape(pred, [-1])

    label_cls, _ = tf.unique(label_ravel)
    pred_cls, _ = tf.unique(pred_ravel)
    nonintersect_cls_n = tf.abs(
        tf.subtract(tf.shape(label_cls), tf.shape(pred_cls)))

    intersection = tf.math.confusion_matrix(label_ravel, pred_ravel)
    area_label = tf.reduce_sum(intersection, axis=0)
    area_pred = tf.reduce_sum(intersection, axis=1)

    union = area_label + area_pred - intersection

    intersection = tf.cast(intersection, tf.float32)
    union = tf.cast(union, tf.float32)

    iou = intersection / (union + 1e-6)

    loss = 0.
    for thr in thresholds:
        matches = tf.greater(iou, thr)
        matches = tf.cast(matches, tf.int32)

        true_positives = tf.cast(
            tf.equal(tf.reduce_sum(matches, axis=1), 1), tf.int32)
        false_positives = tf.cast(
            tf.equal(tf.reduce_sum(matches, axis=1), 0), tf.int32)
        false_negatives = tf.cast(
            tf.equal(tf.reduce_sum(matches, axis=0), 0), tf.int32)

        tp, fp, fn = (
            tf.reduce_sum(true_positives),
            tf.reduce_sum(false_positives),
            tf.reduce_sum(false_negatives)
        )

        precision = tf.cast(
            tf.divide(tp, tp + fp + fn - nonintersect_cls_n), tf.float32)

        loss += precision[0]

    return tf.divide(loss, n_thr)


def average_precision(y_true, y_pred, reduce_mean=True):
    """
    Competition metric.

    Inputs:
        y_true: tf.Tensor of shape [B, H, W, 1] - ground-truth instances
        y_pred: tf.Tensor of shape [B, H, W, 1] - predicted instances
    """
    config = {
        'fn_output_signature': tf.float32
    }

    loss = tf.map_fn(
        average_precision_no_batch, [y_true, y_pred], **config)

    if reduce_mean:
        loss = tf.reduce_mean(loss)

    return loss


def iou(y_true, y_pred, from_logits=True, channel=0):
    """IoU metric for concrete channel (for multi-channel classification)"""
    # y_pred = _preprocess_input(y_pred)

    if from_logits:
        y_pred = tf.nn.sigmoid(y_pred)

    y_true = tf.cast(y_true, tf.float32)

    y_true = y_true[..., channel]
    y_pred = y_pred[..., channel]

    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true * y_pred)
    true_area = tf.reduce_sum(y_true)
    pred_area = tf.reduce_sum(y_pred)

    return tf.divide(
        intersection, true_area + pred_area)


class IoU(MeanMetricWrapper):
    def __init__(
        self,
        from_logits=True,
        channel=0,
        name='IoU'
    ):
        super().__init__(
            iou,
            name=name,
            from_logits=from_logits,
            channel=channel
        )


class ClassPrecision(Precision):
    def __init__(self, class_num=0, **kwargs):
        self.class_num = class_num

        super().__init__(**kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.nn.softmax(y_pred, axis=-1)

        y_true = tf.where(y_true == self.class_num, 1, 0)
        y_pred = y_pred[..., self.class_num]

        return super().update_state(y_true, y_pred, sample_weight=sample_weight)


class ClassRecall(Recall):
    def __init__(self, class_num=0, **kwargs):
        self.class_num = class_num

        super().__init__(**kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.nn.softmax(y_pred, axis=-1)

        y_true = tf.where(y_true == self.class_num, 1, 0)
        y_pred = y_pred[..., self.class_num]

        return super().update_state(y_true, y_pred, sample_weight=sample_weight)
