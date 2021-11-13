import tensorflow as tf
import pathlib

AUTOTUNE = tf.data.experimental.AUTOTUNE


def train_dataset(data_root, image_size, normalize=False, batch_size=128, shuffle=True):
    return load_dataset(data_root, image_size=image_size, label_to_index=None, normalize=normalize,
                        batch_size=batch_size, shuffle=shuffle, with_label=True)


def val_dataset(data_root, image_size, label_to_index, normalize=False, batch_size=128, shuffle=True):
    return load_dataset(data_root, image_size=image_size, label_to_index=label_to_index, normalize=normalize,
                        batch_size=batch_size, shuffle=shuffle, with_label=True)


def infer_dataset(data_root, image_size, normalize=False, batch_size=128, shuffle=True):
    return load_dataset(data_root, image_size=image_size, label_to_index=None, normalize=normalize,
                        batch_size=batch_size, shuffle=shuffle, with_label=False)


def load_dataset(data_root,
                 image_size, label_to_index=None,
                 normalize=True, batch_size=128, 
                 shuffle=True, with_label=True):
    
    data_root = pathlib.Path(data_root)
    if with_label:
        all_image_paths = list(data_root.glob('*/*'))
        if label_to_index is None:
            label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
            label_to_index = dict((name, index) for index, name in enumerate(label_names))
            
        all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                            for path in all_image_paths]

        ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
        
        ds = ds.map(lambda image_path, image_label:
                    load_and_preprocess_from_path_label(image_path, image_label, image_size, normalize))
    else:
        all_image_paths = list(data_root.glob('*/'))
        ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
        ds = ds.map(lambda image_path: load_and_preprocess_image(image_path, image_size, normalize))

    if shuffle:
        image_count = len(all_image_paths)
        ds = ds.shuffle(buffer_size=image_count)
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return label_to_index, ds


def preprocess_image(image, image_size, normalize=False):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, image_size)
    if normalize:
        image /= 255.0
    return image

    
def load_and_preprocess_image(path, image_size, normalize=False):
    image = tf.io.read_file(path)
    return preprocess_image(image, image_size, normalize=normalize)


def load_and_preprocess_from_path_label(path, label, image_size, normalize=False):
    return load_and_preprocess_image(path, image_size, normalize), label


class DataAugmentation(tf.keras.layers.Layer):
    def __init__(self,  rotation_range=10, shift=(0.1, 0.1), shear=10, zoom=(0.1, 0.1)):
        super().__init__(name='patch_embed')
        self.rotation_range = rotation_range
        self.shift = shift
        self.shear = shear
        self.zoom = zoom

    def call(self, x):
        x = tf.keras.preprocessing.image.random_rotation(x, self.rotation_range, row_axis=0, col_axis=1,
                                                         channel_axis=2)
        x = tf.keras.preprocessing.image.random_shift(x, *self.random_shift, row_axis=0, col_axis=1,
                                                      channel_axis=2)
        x = tf.keras.preprocessing.image.random_shear(x,  self.shear, row_axis=0, col_axis=1,
                                                      channel_axis=2)
        x = tf.keras.preprocessing.image.random_zoom(x, *self.zoom, row_axis=0, col_axis=1,
                                                     channel_axis=2)
        return x

