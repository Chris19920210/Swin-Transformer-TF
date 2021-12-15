import tensorflow as tf
from data_processing.autoaugment import distort_image_with_autoaugment, distort_image_with_randaugment
import pathlib

CROP_PADDING = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE


def preprocess_image(image_bytes, image_size, is_training=False
                     , augment_name=None,
                     randaug_num_layers=None, randaug_magnitude=None):
    """Preprocesses the given image.
    Args:
      image_bytes: `Tensor` representing an image binary of arbitrary size.
      is_training: `bool` for whether the preprocessing is for training.
      use_bfloat16: `bool` for whether to use bfloat16.
      image_size: image size.
      augment_name: `string` that is the name of the augmentation method
        to apply to the image. `autoaugment` if AutoAugment is to be used or
        `randaugment` if RandAugment is to be used. If the value is `None` no
        augmentation method will be applied applied. See autoaugment.py for more
        details.
      randaug_num_layers: 'int', if RandAug is used, what should the number of
        layers be. See autoaugment.py for detailed description.
      randaug_magnitude: 'int', if RandAug is used, what should the magnitude
        be. See autoaugment.py for detailed description.
    Returns:
      A preprocessed image `Tensor` with value range of [0, 255].
    """
    if is_training:
        return tf.keras.applications.imagenet_utils.preprocess_input(preprocess_for_train(
            image_bytes, image_size, augment_name,
            randaug_num_layers, randaug_magnitude),
            mode="torch")
    else:
        return tf.keras.applications.imagenet_utils.preprocess_input(preprocess_for_eval(image_bytes, image_size),
                                                                     mode="torch")


def distorted_bounding_box_crop(image_bytes,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100):
    """Generates cropped_image using one of the bboxes randomly distorted.
    See `tf.image.sample_distorted_bounding_box` for more documentation.
    Args:
      image_bytes: `Tensor` of binary image data.
      bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
          where each coordinate is [0, 1) and the coordinates are arranged
          as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
          image.
      min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
          area of the image must contain at least this fraction of any bounding
          box supplied.
      aspect_ratio_range: An optional list of `float`s. The cropped area of the
          image must have an aspect ratio = width / height within this range.
      area_range: An optional list of `float`s. The cropped area of the image
          must contain a fraction of the supplied image within in this range.
      max_attempts: An optional `int`. Number of attempts at generating a cropped
          region of the image of the specified constraints. After `max_attempts`
          failures, return the entire image.
      scope: Optional `str` for name scope.
    Returns:
      cropped image `Tensor`
    """
    with tf.name_scope('distorted_bounding_box_crop'):
        shape = tf.image.extract_jpeg_shape(image_bytes)
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            shape,
            bounding_boxes=bbox,
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, _ = sample_distorted_bounding_box

        # Crop the image to the specified bounding box.
        offset_y, offset_x, _ = tf.unstack(bbox_begin)
        target_height, target_width, _ = tf.unstack(bbox_size)
        crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
        image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)

        return image


def _at_least_x_are_equal(a, b, x):
    """At least `x` of `a` and `b` `Tensors` are equal."""
    match = tf.equal(a, b)
    match = tf.cast(match, tf.int32)
    return tf.greater_equal(tf.reduce_sum(match), x)


def _decode_and_random_crop(image_bytes, image_size):
    """Make a random crop of image_size."""
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    image = distorted_bounding_box_crop(
        image_bytes,
        bbox,
        min_object_covered=0.1,
        aspect_ratio_range=(3. / 4, 4. / 3.),
        area_range=(0.08, 1.0),
        max_attempts=10)
    original_shape = tf.image.extract_jpeg_shape(image_bytes)
    bad = _at_least_x_are_equal(original_shape, tf.shape(image), 3)

    image = tf.cond(
        bad,
        lambda: _decode_and_center_crop(image_bytes, image_size),
        lambda: tf.image.resize([image], [image_size, image_size], method="bicubic")[0])

    return image


def _decode_and_center_crop(image_bytes, image_size):
    """Crops to center of image with padding then scales image_size."""
    shape = tf.image.extract_jpeg_shape(image_bytes)
    image_height = shape[0]
    image_width = shape[1]

    padded_center_crop_size = tf.cast(
        ((image_size / (image_size + CROP_PADDING)) *
         tf.cast(tf.minimum(image_height, image_width), tf.float32)),
        tf.int32)

    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2
    crop_window = tf.stack([offset_height, offset_width,
                            padded_center_crop_size, padded_center_crop_size])
    image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
    image = tf.image.resize([image], [image_size, image_size], method="bicubic")[0]

    return image


def _flip(image):
    """Random horizontal image flip."""
    image = tf.image.random_flip_left_right(image)
    return image


def preprocess_for_eval(image_bytes, image_size):
    """Preprocesses the given image for evaluation.
    Args:
      image_bytes: `Tensor` representing an image binary of arbitrary size.
      use_bfloat16: `bool` for whether to use bfloat16.
      image_size: image size.
    Returns:
      A preprocessed image `Tensor`.
    """
    image = _decode_and_center_crop(image_bytes, image_size)
    image = tf.reshape(image, [image_size, image_size, 3])
    return image


def preprocess_for_train(image_bytes, image_size,
                         augment_name=None,
                         randaug_num_layers=None, randaug_magnitude=None):
    """Preprocesses the given image for evaluation.
    Args:
      image_bytes: `Tensor` representing an image binary of arbitrary size.
      use_bfloat16: `bool` for whether to use bfloat16.
      image_size: image size.
      augment_name: `string` that is the name of the augmentation method
        to apply to the image. `autoaugment` if AutoAugment is to be used or
        `randaugment` if RandAugment is to be used. If the value is `None` no
        augmentation method will be applied applied. See autoaugment.py for more
        details.
      randaug_num_layers: 'int', if RandAug is used, what should the number of
        layers be. See autoaugment.py for detailed description.
      randaug_magnitude: 'int', if RandAug is used, what should the magnitude
        be. See autoaugment.py for detailed description.
    Returns:
      A preprocessed image `Tensor`.
    """
    image = _decode_and_random_crop(image_bytes, image_size)
    image = _flip(image)
    image = tf.reshape(image, [image_size, image_size, 3])

    if augment_name:
        tf.compat.v1.logging.info('Apply AutoAugment policy %s', augment_name)
        input_image_type = image.dtype
        image = tf.clip_by_value(image, 0.0, 255.0)
        image = tf.cast(image, dtype=tf.uint8)

        if augment_name == 'autoaugment':
            image = distort_image_with_autoaugment(image, 'v0')
        elif augment_name == 'randaugment':
            image = distort_image_with_randaugment(
                image, randaug_num_layers, randaug_magnitude)
        else:
            raise ValueError('Invalid value for augment_name: %s' % (augment_name))

        image = tf.cast(image, dtype=input_image_type)
    return image


def train_dataset(data_root, image_size, is_training=True,
                  augment_name='autoaugment', batch_size=128, shuffle=True):
    return load_dataset(data_root, image_size=image_size, label_to_index=None, is_training=is_training,
                        augment_name=augment_name, batch_size=batch_size, shuffle=shuffle, with_label=True)


def val_dataset(data_root, image_size, label_to_index, task1_to_task2, label_to_index_task2, batch_size=128,
                shuffle=False, tracer=False):
    return load_dataset(data_root, image_size=image_size, label_to_index=label_to_index, task1_to_task2=task1_to_task2,
                        label_to_index_task2=label_to_index_task2, is_training=False,
                        batch_size=batch_size, shuffle=shuffle, with_label=True, repeat=False, tracer=tracer)


def infer_dataset(data_root, image_size, batch_size=128, shuffle=True):
    return load_dataset(data_root, image_size=image_size, label_to_index=None, is_training=False,
                        batch_size=batch_size, shuffle=shuffle, with_label=False, repeat=False)


def load_dataset(data_root,
                 image_size, label_to_index=None, task1_to_task2=None, label_to_index_task2=None,
                 is_training=False, augment_name=None, batch_size=128,
                 shuffle=True, with_label=True, repeat=True, tracer=False):
    data_root = pathlib.Path(data_root)
    if with_label:
        all_image_paths = list(data_root.glob('*/*'))
        all_image_paths = [str(path) for path in all_image_paths]

        if label_to_index is None or task1_to_task2 is None or label_to_index_task2 is None:
            label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
            task1_to_task2 = {label_name: label_name.split(" ")[0] for label_name in label_names}
            task2_names = set(task1_to_task2.values())
            label_to_index_task2 = dict((name, index) for index, name in enumerate(task2_names))
            label_to_index = dict((name, index) for index, name in enumerate(label_names))

        all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                            for path in all_image_paths]

        all_image_labels_task2 = [label_to_index_task2[task1_to_task2[pathlib.Path(path).parent.name]]
                                  for path in all_image_paths]

        ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels, all_image_labels_task2))

        if not tracer:
            ds = ds.map(lambda image_path, image_label, image_label_task2:
                        load_and_preprocess_from_path_label(image_path, image_label, image_label_task2, image_size,
                                                            is_training=is_training,
                                                            augment_name=augment_name))
        else:
            ds = ds.map(lambda image_path, image_label, image_label_task2:
                        (*load_and_preprocess_from_path_label(image_path, image_label, image_label_task2, image_size,
                                                              is_training=is_training,
                                                              augment_name=augment_name), image_path))
    else:
        all_image_paths = list(data_root.glob('*/'))
        all_image_paths = [str(path) for path in all_image_paths]
        ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
        if not tracer:
            ds = ds.map(lambda image_path: load_and_preprocess_image(image_path, image_size, is_training=is_training,
                                                                     augment_name=augment_name))
        else:
            ds = ds.map(lambda image_path: (load_and_preprocess_image(image_path, image_size, is_training=is_training,
                                                                      augment_name=augment_name), image_path))

    if shuffle:
        image_count = len(all_image_paths)
        ds = ds.shuffle(buffer_size=image_count)
    if repeat:
        ds = ds.repeat()
    else:
        ds = ds.repeat(1)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return len(all_image_paths), label_to_index, task1_to_task2, label_to_index_task2, ds


def load_and_preprocess_image(path, image_size, is_training=True, augment_name='autoaugment'):
    image = tf.io.read_file(path)
    return preprocess_image(image, image_size, is_training=is_training, augment_name=augment_name)


def load_and_preprocess_from_path_label(path, label, image_label_task2, image_size, is_training=True,
                                        augment_name='autoaugment'):
    return load_and_preprocess_image(path, image_size, is_training=is_training,
                                     augment_name=augment_name), label, image_label_task2
