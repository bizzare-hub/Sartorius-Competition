import os
import glob
import logging
import re

import tensorflow as tf


class KeepMaxCheckpoints(tf.keras.callbacks.Callback):
    def __init__(self, keep_max, ckpt_dir):
        super().__init__()

        self.keep_max = keep_max
        self.ckpt_dir = ckpt_dir

    def remove_ckpt(self, idx):
        index_path = os.path.join(
            self.ckpt_dir, f'model_{idx}.index')
        data_path = os.path.join(
            self.ckpt_dir, f'model_{idx}.data-00000-of-00001')

        os.remove(index_path)
        os.remove(data_path)

    def on_epoch_end(self, epoch, logs=None):
        files = glob.glob(os.path.join(self.ckpt_dir, '*'))

        matches = [re.findall(r'model_\d+', f) for f in files]
        model_indices = {
            int(match[0][6:]) for match in matches if len(match) > 0}
        model_indices = list(map(str, (sorted(model_indices))))

        length = len(model_indices)
        to_remove_length = length - self.keep_max

        if to_remove_length > 0:
            indices_to_remove = model_indices[:to_remove_length]

            for idx in indices_to_remove:
                self.remove_ckpt(idx)
