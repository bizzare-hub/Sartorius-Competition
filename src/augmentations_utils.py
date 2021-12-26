import albumentations as A
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


@tf.function
def _get_box(img_h, img_w, ratio):
    cut_w = img_w * ratio
    cut_w = tf.cast(cut_w, tf.int32)

    cut_h = img_h * ratio
    cut_h = tf.cast(cut_h, tf.int32)

    cut_x = tf.random.uniform((1,), minval=0, maxval=img_w, dtype=tf.int32)
    cut_y = tf.random.uniform((1,), minval=0, maxval=img_h, dtype=tf.int32)

    x_min = tf.clip_by_value(cut_x[0] - cut_w // 2, 0, img_w)
    y_min = tf.clip_by_value(cut_y[0] - cut_h // 2, 0, img_h)
    x_max = tf.clip_by_value(cut_x[0] + cut_w // 2, 0, img_w)
    y_max = tf.clip_by_value(cut_y[0] + cut_h // 2, 0, img_h)

    target_h = y_max - y_min
    if target_h == 0:
        target_h += 1

    target_w = x_max - x_min
    if target_w == 0:
        target_w += 1

    return x_min[0], y_min[0], target_h[0], target_w[0]


@tf.function
def tf_cutmix(data1, data2, min_ratio=0.05, max_ratio=0.25):
    (image1, mask1), type1 = data1
    (image2, mask2), type2 = data2

    if type1 != type2:
        return image1, mask1

    img_h, img_w, _ = image1.get_shape().as_list()

    cut_ratio = tf.random.uniform(shape=(1,), minval=min_ratio, maxval=max_ratio)

    box_x, box_y, box_h, box_w = _get_box(img_h, img_w, cut_ratio)

    patch_mask = tf.ones(shape=[box_h, box_w, 1], dtype=tf.float32)
    patch_mask = tf.image.pad_to_bounding_box(
        patch_mask, box_y, box_x, img_h, img_w)

    patch_mask = tf.cast(patch_mask, image1.dtype)
    image = tf.multiply(image1, 1 - patch_mask)
    image = image + tf.multiply(image2, patch_mask)

    patch_mask = tf.cast(patch_mask, mask1.dtype)
    mask = tf.multiply(mask1, 1 - patch_mask)
    mask = mask + tf.multiply(mask2, patch_mask)

    return image, mask


@tf.function
def tf_gaussian_noise(img, mean=0.0, stddev=1.0):
    h, w, c = img.get_shape().as_list()

    noise = tf.random.normal(
        [h, w, c], mean=mean, stddev=stddev)

    return img + noise


@tf.function
def tf_gaussian_blur(img, filter_limit=5, sigma_limit=1.1):
    filters = tf.range(start=1, limit=filter_limit + 2, delta=2)
    filter_shape = tf.random.shuffle(filters)[0]

    sigma = tf.random.uniform([], 0., sigma_limit, tf.float32)

    return tfa.image.gaussian_filter2d(
        img,
        filter_shape=filter_shape,
        sigma=sigma
    )


@tf.function
def tf_median_blur(img, filter_limit=5):
    filters = tf.range(start=1, limit=filter_limit + 2, delta=2)
    filter_shape = tf.random.shuffle(filters)[0]

    return tfa.image.median_filter2d(
        img,
        filter_shape=tf.cast(filter_shape, tf.int32)
    )


def albu_grid_distortion(img, mask, num_steps, distort_limit):
    aug_cfg = {
        'num_steps': num_steps,
        'distort_limit': float(distort_limit),
        'p': 1.0
    }

    aug = A.GridDistortion(**aug_cfg)

    mask = mask.astype(np.float32)  # WHYYY????

    params = aug.get_params()

    return aug.apply(img, **params),\
           aug.apply_to_mask(mask, **params).astype(np.int32)


@tf.function
def tf_grid_distortion(img, mask, num_steps=5, distort_limit=0.05):
    return tf.numpy_function(albu_grid_distortion,
                             [img, mask, num_steps, distort_limit],
                             (tf.float32, tf.int32))


def albu_elastic_transform(img, mask, alpha, sigma, alpha_affine):
    aug_cfg = {
        'alpha': int(alpha),
        'sigma': sigma,
        'alpha_affine': alpha_affine,
        'p': 1.0
    }
    aug = A.ElasticTransform(**aug_cfg)

    mask = mask.astype(np.float32)  # WHYYY????

    params = aug.get_params()

    return aug.apply(img, **params),\
           aug.apply_to_mask(mask, **params).astype(np.int32)


@tf.function
def tf_elastic_transform(img, mask, alpha=1, sigma=50, alpha_affine=50):
    return tf.numpy_function(albu_elastic_transform,
                             [img, mask, alpha, sigma, alpha_affine],
                             (tf.float32, tf.int32))
