# Swin Transformer (Tensorflow)
Tensorflow reimplementation of **Swin Transformer** model.   
  
Based on [Official Pytorch implementation](https://github.com/microsoft/Swin-Transformer).
![image](https://user-images.githubusercontent.com/24825165/121768619-038e6d80-cb9a-11eb-8cb7-daa827e7772b.png)

## Requirements
- `tensorflow == 2.4.1`

## Pretrained Swin Transformer Checkpoints
**ImageNet-1K and ImageNet-22K Pretrained Checkpoints**  
| name | pretrain | resolution |acc@1 | #params | model |
| :---: | :---: | :---: | :---: | :---: | :---: |
|`swin_tiny_224` |ImageNet-1K |224x224|81.2|28M|[github](https://github.com/rishigami/Swin-Transformer-TF/releases/download/v0.1-tf-swin-weights/swin_tiny_224.tgz)|
|`swin_small_224`|ImageNet-1K |224x224|83.2|50M|[github](https://github.com/rishigami/Swin-Transformer-TF/releases/download/v0.1-tf-swin-weights/swin_small_224.tgz)|
|`swin_base_224` |ImageNet-22K|224x224|85.2|88M|[github](https://github.com/rishigami/Swin-Transformer-TF/releases/download/v0.1-tf-swin-weights/swin_base_224.tgz)|
|`swin_base_384` |ImageNet-22K|384x384|86.4|88M|[github](https://github.com/rishigami/Swin-Transformer-TF/releases/download/v0.1-tf-swin-weights/swin_base_384.tgz)|
|`swin_large_224`|ImageNet-22K|224x224|86.3|197M|[github](https://github.com/rishigami/Swin-Transformer-TF/releases/download/v0.1-tf-swin-weights/swin_large_224.tgz)|
|`swin_large_384`|ImageNet-22K|384x384|87.3|197M|[github](https://github.com/rishigami/Swin-Transformer-TF/releases/download/v0.1-tf-swin-weights/swin_large_384.tgz)|

## Inference Demo
```python
import tensorflow as tf
import pickle as pkl
from PIL import Image
import numpy as np

## show demo image
image = Image.open('topic_model/demo.jpeg')
image.show()

## load model
model = tf.keras.models.load_model('topic_model/')

## load image
IMAGE_SIZE = 224
CROP_PADDING = 32
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

def load_and_preprocess_image(path, image_size):
    image = tf.io.read_file(path)
    return preprocess_for_eval(image, image_size)

image = load_and_preprocess_image('topic_model/demo.jpeg', image_size=IMAGE_SIZE)
## with batch size
image = tf.expand_dims(image, axis=0)

## load label
label_to_index = pkl.load(open('topic_model/label_to_index.pkl', 'rb'))
labels = sorted(label_to_index, key=lambda key: label_to_index[key])

## predict
result = model.predict(image)
label = np.argmax(result, axis=-1)
print(labels[label[0]])
```

## Examples
Initializing the model:
```python
from swintransformer import SwinTransformer

model = SwinTransformer('swin_tiny_224', num_classes=1000, include_top=True, pretrained=False)
```
You can use a pretrained model like this:
```python
import tensorflow as tf
from swintransformer import SwinTransformer

model = tf.keras.Sequential([
  tf.keras.layers.Lambda(lambda data: tf.keras.applications.imagenet_utils.preprocess_input(tf.cast(data, tf.float32), mode="torch"), input_shape=[*IMAGE_SIZE, 3]),
  SwinTransformer('swin_tiny_224', include_top=False, pretrained=True),
  tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])
```
If you use a pretrained model with TPU on kaggle, specify `use_tpu` option:
```python
import tensorflow as tf
from swintransformer import SwinTransformer

model = tf.keras.Sequential([
  tf.keras.layers.Lambda(lambda data: tf.keras.applications.imagenet_utils.preprocess_input(tf.cast(data, tf.float32), mode="torch"), input_shape=[*IMAGE_SIZE, 3]),
  SwinTransformer('swin_tiny_224', include_top=False, pretrained=True, use_tpu=True),
  tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])
```
Example: [TPU training on Kaggle](https://www.kaggle.com/rishigami/tpu-swin-transformer-tensorflow)
## Citation
```
@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}

