from functools import partial
import numpy as np
import tensorflow as tf

from .augmentations_utils import (
    tf_cutmix, tf_gaussian_noise,
    tf_gaussian_blur, tf_median_blur,
    tf_grid_distortion, tf_elastic_transform
)


class Augmentation:
    def __init__(self, name, fn, params, apply_mask, prob) -> None:
        self.fn = partial(fn, **params) if params else fn
        self.prob = prob
        self.name = name
        self.apply_mask = apply_mask

    def __call__(self, *images):
        x = tf.cond(
            tf.random.uniform([], 0, 1, tf.float32) <= self.prob,
            lambda: tuple([self.fn(image) for image in images]),
            lambda: images
        )

        return x


class OneOf(Augmentation):
    def __init__(self, *augmentations, prob) -> None:
        self.augmentations = augmentations

        aug_probs = [aug.prob for aug in augmentations]
        self.scaled_probs = [p / sum(aug_probs) for p in aug_probs]

        for aug in self.augmentations:
            aug.prob = 1.  # oneof class will choose which aug. to apply

        apply_mask = True if self.augmentations[0].apply_mask else False

        super().__init__('OneOf', lambda x: x, {}, apply_mask, prob)

    def _apply_aug(self, images):
        aug_idx = tf.argmin(
            tf.where(self.scaled_probs >= tf.random.uniform([], 0, 1, tf.float32)))[0]
        augmentation = tf.gather(self.augmentations, aug_idx)

        x = tf.cond(
            tf.cast(augmentation.name in ['ElasticTransform', 'GridDistortion'], tf.bool),
            lambda: augmentation(images[0], images[1]),
            lambda: tuple([augmentation(image) for image in images])
        )

        return x

    def __call__(self, *images):
        x = tf.cond(
            tf.random.uniform([], 0, 1, tf.float32) <= self.prob,
            lambda: self._apply_aug(images),
            lambda: images
        )

        return x


class RandomBrigthness(Augmentation):
    def __init__(self, max_delta, prob) -> None:
        params = {
            'max_delta': max_delta
        }

        super().__init__('RandomBrigthness',
                         tf.image.random_brightness,
                         params, False, prob)


class RandomContrast(Augmentation):
    def __init__(self, lower, upper, prob) -> None:
        params = {
            'lower': lower,
            'upper': upper
        }

        super().__init__('RandomContrast',
                         tf.image.random_contrast,
                         params, False, prob)


class RandomCrop(Augmentation):
    def __init__(self, crop_shape, prob) -> None:
        self.crop_shape = crop_shape

        super().__init__('RandomCrop',
                         tf.image.crop_to_bounding_box,
                         None, True, prob)

    def __call__(self, *images):
        shape = tf.shape(images[0])[:2]
        dh, dw = tf.unstack(shape - self.crop_shape + 1)

        h = tf.random.uniform(shape=[], maxval=dh, dtype=tf.int32)
        w = tf.random.uniform(shape=[], maxval=dw, dtype=tf.int32)

        crop_fn = partial(self.fn,
                          offset_height=h,
                          offset_width=w,
                          target_height=self.crop_shape[0],
                          target_width=self.crop_shape[1])

        x = tf.cond(
            tf.random.uniform([], 0, 1, tf.float32) <= self.prob,
            lambda: tuple([crop_fn(image) for image in images]),
            lambda: images
        )

        return x


class RandomLeftRightFlip(Augmentation):
    def __init__(self, prob) -> None:
        super().__init__('RandomLeftRightFlip',
                         tf.image.flip_left_right,
                         None, True, prob)


class RandomUpDownFlip(Augmentation):
    def __init__(self, prob) -> None:
        super().__init__('RandomUpDownFlip',
                         tf.image.flip_up_down,
                         None, True, prob)


class CutMix(Augmentation):
    def __init__(self, min_ratio, max_ratio, prob) -> None:
        params = {
            'min_ratio': min_ratio,
            'max_ratio': max_ratio
        }

        super().__init__(
            'CutMix', tf_cutmix, params, True, prob)

    def __call__(self, data1, data2):
        images = tf.cond(
            tf.random.uniform([], 0, 1, tf.float32) > 1 - self.prob,
            lambda: self.fn(data1, data2),
            lambda: data1[0]
        )

        return images


class Transpose(Augmentation):
    def __init__(self, prob) -> None:
        params = {
            'perm': [1, 0, 2]
        }

        super().__init__(
            'Transpose', tf.transpose, params, True, prob)


class RandomGaussianNoise(Augmentation):
    def __init__(self, mean, stddev, prob) -> None:
        params = {
            'mean': mean,
            'stddev': stddev
        }

        super().__init__(
            'RandomGaussianNoise', tf_gaussian_noise, params, False, prob)


# TODO: FIX BLUR PROBLEMS (CURRENTLY THEY'RE NOT WORKING)
class RandomGaussianBlur(Augmentation):
    def __init__(self, filter_limit, sigma_limit, prob) -> None:
        params = {
            'filter_limit': filter_limit,
            'sigma_limit': sigma_limit
        }

        super().__init__(
            'RandomGaussianBlur', tf_gaussian_blur, params, False, prob)


class RandomMedianBlur(Augmentation):
    def __init__(self, filter_limit, prob) -> None:
        params = {
            'filter_limit': filter_limit
        }

        super().__init__(
            'RandomMedianBlur', tf_median_blur, params, False, prob)


class GridDistortion(Augmentation):
    def __init__(self, num_steps, distort_limit, prob) -> None:
        params = {
            'num_steps': num_steps,
            'distort_limit': distort_limit
        }

        super().__init__(
            'GridDistortion', tf_grid_distortion, params, True, prob)

    def __call__(self, image, mask):
        img_shape = image.get_shape().as_list()
        mask_shape = mask.get_shape().as_list()

        x = tf.cond(
            tf.random.uniform([], 0, 1, tf.float32) <= self.prob,
            lambda: tuple(self.fn(image, mask)),
            lambda: tuple((image, mask))
        )

        x = tuple([tf.reshape(x[0], img_shape),
                   tf.reshape(x[1], mask_shape)])

        return x


class ElasticTransform(Augmentation):
    def __init__(self, alpha, sigma, alpha_affine, prob) -> None:
        params = {
            'alpha': alpha,
            'sigma': sigma,
            'alpha_affine': alpha_affine
        }

        super().__init__(
            'ElasticTransform', tf_elastic_transform, params, True, prob)

    def __call__(self, image, mask):
        img_shape = image.get_shape().as_list()
        mask_shape = mask.get_shape().as_list()

        x = tf.cond(
            tf.random.uniform([], 0, 1, tf.float32) <= self.prob,
            lambda: tuple(self.fn(image, mask)),
            lambda: tuple((image, mask))
        )

        x = tuple([tf.reshape(x[0], img_shape),
                   tf.reshape(x[1], mask_shape)])

        return x
