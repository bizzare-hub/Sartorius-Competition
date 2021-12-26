import math
from pathlib import Path
from PIL import Image

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE

from .dataset_utils import apply_augmentations


def _read_image_opencv(path):
    return cv2.imread(
        path.decode('UTF-8'),
        cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR
    )


def _read_image_numpy(path):
    image = np.array(
        Image.open(path.decode('UTF-8')))

    return image


@tf.function
def _read_image(path, parse_shape=(520, 704, 1), dtype=tf.uint8):
    image = tf.numpy_function(_read_image_opencv, [path], dtype)
    image = tf.reshape(image, parse_shape)

    return image


@tf.function
def _read_image_v2(path, parse_shape=(520, 704, 1), dtype=tf.uint8):
    image = tf.numpy_function(_read_image_numpy, [path], dtype)
    image = tf.reshape(image, parse_shape)

    return image


@tf.function
def _resize_image(image, resize_shape=(512, 512)):
    return tf.image.resize(image, resize_shape, method='nearest')


class SartoriusDataset:
    domains_path_mapping = {
        'images': ('', '.png'),
        'masks': ('_mask', '.tif'),
        'masks_w_contours': ('_mask_w_contour', '.tif'),
        'submasks_w_contours': ('_mask_w_contour', '.tif'),
        'masks_w_intersections': ('_mask_w_intersection', '.tif'),
        'categorical': ('_categorical_mask', '.tif'),
        'categorical_intersection': ('_categorical_intersection', '.tif'),
        'dsb2018_masks': ('_dsb2018_mask', '.tif'),
        'dsb2018_masks_v2': ('_dsb2018_mask', '.tif'),
        'dsb2018_categorical': ('_dsb2018_mask', '.tif')
    }

    domains_parse_mapping = {
        'images': _read_image,
        'masks': _read_image,
        'masks_w_contours': _read_image_v2,
        'submasks_w_contours': _read_image_v2,
        'masks_w_intersections': _read_image_v2,
        'categorical': _read_image,
        'categorical_intersection': _read_image,
        'dsb2018_masks': _read_image_v2,
        'dsb2018_masks_v2': _read_image_v2,
        'dsb2018_categorical': _read_image
    }

    def __init__(
        self,
        root_dir: Path,
        subdirs: list = None,
        input_domain: str = 'images',
        target_domain: str = 'masks',
        types_df: pd.DataFrame = None,  # TODO: Add subdirs equivalent for types_df
        batch_size: int = 16,
        image_parse_shape: tuple = (520, 704, 1),
        mask_parse_shape: tuple = (520, 704, 1),
        resize_shape: tuple = None,
        augmentations: list = None,
        shuffle: bool = False,
        shuffle_buffer_size: int = None,
        random_state: int = None
    ) -> None:

        self.input_domain = input_domain
        self.target_domain = target_domain
        self.image_parse_shape = image_parse_shape
        self.mask_parse_shape = mask_parse_shape
        self.resize_shape = resize_shape
        self.augmentations = augmentations or []
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.batch_size = batch_size
        self.seed = random_state

        parse_dirs = [root_dir] + subdirs if subdirs is not None else [root_dir]

        img_paths, mask_paths = self.parse_paths(parse_dirs)

        self.cell_types = None
        if types_df is not None:
            ids = [Path(p).stem for p in img_paths]
            cell_types = types_df.set_index('id')
            self.cell_types = cell_types.loc[ids].type.values.tolist()

        dataset = tuple(tf.data.Dataset.from_tensor_slices(p)
                        for p in [img_paths, mask_paths])
        dataset = tf.data.Dataset.zip(dataset)

        dataset = self.shuffle_dataset(dataset)
        dataset = self.parse_dataset(dataset)
        dataset = self.preprocess_dataset(dataset)
        dataset = self.batch_dataset(dataset)

        self.dataset = dataset
        self.n_steps = int(math.ceil(len(img_paths) / batch_size))

    def parse_paths(self, dirs: list):
        input_patt, input_ext = self.domains_path_mapping[self.input_domain]
        target_patt, target_ext = self.domains_path_mapping[self.target_domain]

        input_stem = input_patt + input_ext
        target_stem = target_patt + target_ext

        all_input_paths = []
        all_target_paths = []

        for dir in dirs:
            input_dir = dir/self.input_domain
            input_paths = list(input_dir.glob(f'*{input_stem}'))
            input_paths = list(map(lambda p: str(p), input_paths))

            target_dir = dir/self.target_domain
            target_paths = [str(target_dir/Path(p).name).replace(input_stem, target_stem)
                            for p in input_paths]

            all_input_paths.extend(input_paths)
            all_target_paths.extend(target_paths)

        return all_input_paths, all_target_paths

    def shuffle_dataset(self, dataset: tf.data.Dataset):
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size)
        return dataset

    def parse_dataset(self, dataset: tf.data.Dataset):
        def parse(image_path, mask_path):
            image = self.domains_parse_mapping[self.input_domain](
                image_path, self.image_parse_shape)
            image = tf.cast(image, tf.float32)

            mask = self.domains_parse_mapping[self.target_domain](
                mask_path, self.mask_parse_shape, dtype=tf.uint8)
            mask = tf.cast(mask, tf.int32)

            return image, mask

        return dataset.map(parse, num_parallel_calls=AUTOTUNE)

    def preprocess_dataset(self, dataset: tf.data.Dataset):
        dataset = apply_augmentations(
            dataset, self.augmentations, self.cell_types)

        def preprocess(image, mask):
            image = image / 255.
            image = tf.clip_by_value(image, 0, 1)

            if self.resize_shape is not None:
                image = _resize_image(image, self.resize_shape)
                mask = _resize_image(mask, self.resize_shape)

            return image, mask

        return dataset.map(preprocess, num_parallel_calls=AUTOTUNE)

    def batch_dataset(self, dataset: tf.data.Dataset):
        return dataset.batch(self.batch_size)


class SartoriusDatasetV2:
    domains_path_mapping = {
        'images': ('', '.png'),
        'masks': ('_mask', '.tif'),
        'masks_w_contours': ('_mask_w_contour', '.tif'),
        'submasks_w_contours': ('_mask_w_contour', '.tif'),
        'masks_w_intersections': ('_mask_w_intersection', '.tif'),
        'categorical': ('_categorical_mask', '.tif'),
        'categorical_intersection': ('_categorical_intersection', '.tif'),
        'dsb2018_masks': ('_dsb2018_mask', '.tif'),
        'dsb2018_masks_v2': ('_dsb2018_mask', '.tif'),
        'dsb2018_categorical': ('_dsb2018_mask', '.tif')
    }

    domains_parse_mapping = {
        'images': _read_image,
        'masks': _read_image,
        'masks_w_contours': _read_image_v2,
        'submasks_w_contours': _read_image_v2,
        'masks_w_intersections': _read_image_v2,
        'categorical': _read_image,
        'categorical_intersection': _read_image,
        'dsb2018_masks': _read_image_v2,
        'dsb2018_masks_v2': _read_image_v2,
        'dsb2018_categorical': _read_image
    }

    def __init__(
        self,
        root_dir: Path,
        subdirs: list = None,
        input_domain: str = 'images',
        target_domain: str = 'masks',
        types_df: pd.DataFrame = None,
        batch_size: int = 16,
        image_parse_shape: tuple = (520, 704, 1),
        mask_parse_shape: tuple = (520, 704, 1),
        resize_shape: tuple = None,
        augmentations: list = None,
        shuffle: bool = False,
        shuffle_buffer_size: int = None,
        random_state: int = None
    ) -> None:

        self.input_domain = input_domain
        self.target_domain = target_domain
        self.image_parse_shape = image_parse_shape
        self.mask_parse_shape = mask_parse_shape
        self.resize_shape = resize_shape
        self.augmentations = augmentations or []
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.batch_size = batch_size
        self.seed = random_state

        parse_dirs = [root_dir] + subdirs if subdirs is not None else [root_dir]

        img_paths, mask_paths = self.parse_paths(parse_dirs)

        ids = [Path(p).stem for p in img_paths]
        cell_types = types_df.set_index('id')
        self.cell_types = cell_types.loc[ids].type.values.tolist()
        types_dataset = tf.data.Dataset.from_tensor_slices(self.cell_types)

        dataset = tuple(tf.data.Dataset.from_tensor_slices(p)
                        for p in [img_paths, mask_paths])
        dataset = tf.data.Dataset.zip(dataset)
        dataset = tf.data.Dataset.zip((dataset, types_dataset))

        dataset = self.shuffle_dataset(dataset)

        types_dataset = dataset.map(lambda x, y: y, num_parallel_calls=AUTOTUNE)
        dataset = dataset.map(lambda x, y: x, num_parallel_calls=AUTOTUNE)

        dataset = self.parse_dataset(dataset)
        dataset = self.preprocess_dataset(dataset)

        # merging masks with types after preprocessing
        dataset = self.merge_dataset(dataset, types_dataset)
        dataset = self.batch_dataset(dataset)

        self.dataset = dataset
        self.n_steps = int(math.ceil(len(img_paths) / batch_size))

    def parse_paths(self, dirs: list):
        input_patt, input_ext = self.domains_path_mapping[self.input_domain]
        target_patt, target_ext = self.domains_path_mapping[self.target_domain]

        input_stem = input_patt + input_ext
        target_stem = target_patt + target_ext

        all_input_paths = []
        all_target_paths = []

        for dir in dirs:
            input_dir = dir/self.input_domain
            input_paths = list(input_dir.glob(f'*{input_stem}'))
            input_paths = list(map(lambda p: str(p), input_paths))

            target_dir = dir/self.target_domain
            target_paths = [str(target_dir/Path(p).name).replace(input_stem, target_stem)
                            for p in input_paths]

            all_input_paths.extend(input_paths)
            all_target_paths.extend(target_paths)

        return all_input_paths, all_target_paths

    def shuffle_dataset(self, dataset: tf.data.Dataset):
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size)
        return dataset

    def parse_dataset(self, dataset: tf.data.Dataset):
        def parse(image_path, mask_path):
            image = self.domains_parse_mapping[self.input_domain](
                image_path, self.image_parse_shape)
            image = tf.cast(image, tf.float32)

            mask = self.domains_parse_mapping[self.target_domain](
                mask_path, self.mask_parse_shape, dtype=tf.uint8)
            mask = tf.cast(mask, tf.int32)

            return image, mask

        return dataset.map(parse, num_parallel_calls=AUTOTUNE)

    def preprocess_dataset(self, dataset: tf.data.Dataset):
        dataset = apply_augmentations(
            dataset, self.augmentations, self.cell_types)

        def preprocess(image, mask):
            image = image / 255.
            image = tf.clip_by_value(image, 0, 1)

            if self.resize_shape is not None:
                image = _resize_image(image, self.resize_shape)
                mask = _resize_image(mask, self.resize_shape)

            return image, mask

        return dataset.map(preprocess, num_parallel_calls=AUTOTUNE)

    @staticmethod
    def merge_dataset(data1, data2):
        img_data = data1.map(lambda x, y: x, num_parallel_calls=AUTOTUNE)
        mask_data = data1.map(lambda x, y: y, num_parallel_calls=AUTOTUNE)

        dataset = tf.data.Dataset.zip((mask_data, data2))
        dataset = tf.data.Dataset.zip((img_data, dataset))

        return dataset

    def batch_dataset(self, dataset: tf.data.Dataset):
        return dataset.batch(self.batch_size)
