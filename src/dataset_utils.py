import logging
from functools import partial
import tensorflow as tf

logger = logging.getLogger(__name__)

AUTOTUNE = tf.data.experimental.AUTOTUNE


def apply_augmentations(dataset, augmentations, types):
    def aug_wrapper(image, mask, aug):
        if aug.apply_mask:
            image, mask = aug(image, mask)
        else:
            image = aug(image)[0]

        return image, mask

    augs = []

    for aug in augmentations:
        if aug.name == 'CutMix':
            if types is None:
                raise NotImplementedError("CutMix works only when types are available")

            types_dataset = tf.data.Dataset.from_tensor_slices(types)
            dataset = tf.data.Dataset.zip((dataset, types_dataset))

            dataset1 = dataset.shuffle(int(1e4))

            dataset = tf.data.Dataset.zip((dataset, dataset1))
            dataset = dataset.map(aug, num_parallel_calls=AUTOTUNE)
            logger.info(aug.name)
        else:
            augs.append(aug)

    for aug in augs:
        aug_fn = partial(aug_wrapper, aug=aug)

        dataset = dataset.map(aug_fn, num_parallel_calls=AUTOTUNE)
        logger.info(aug.name)

    return dataset
