import tensorflow as tf
import pathlib
import os
import numpy as np
import shutil

flags = tf.app.flags
flags.DEFINE_string('input', './input', 'Directory to input.')
flags.DEFINE_string('output', './output', 'Directory to output. ')
flags.DEFINE_float('ratio', 0.2, 'ratio')
FLAGS = flags.FLAGS


def main(_):
    data_root = pathlib.Path(FLAGS.input)
    os.mkdir(os.path.join(FLAGS.output, "train"))
    os.mkdir(os.path.join(FLAGS.output, "test"))

    all_image_paths = list(data_root.glob('*/*'))

    for item in data_root.glob('*/'):
        if item.is_dir():
            test_dir = os.path.join(FLAGS.output, "test", item.name)
            train_dir = os.path.join(FLAGS.output, "train", item.name)

            if not os.path.exists(test_dir):
                os.makedirs(test_dir)
            if not os.path.exists(train_dir):
                os.makedirs(train_dir)

    for path in all_image_paths:
        input_path = path.absolute()
        parent_name = path.parent.name
        file_name = path.name
        if np.random.uniform() < FLAGS.ratio:
            shutil.copy(input_path, os.path.join(FLAGS.output, "test", parent_name, file_name))
        else:
            shutil.copy(input_path, os.path.join(FLAGS.output, "train", parent_name, file_name))

    print("Split done!")


if __name__ == "__main__":
    main(0)









