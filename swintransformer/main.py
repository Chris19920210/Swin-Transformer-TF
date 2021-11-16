import tensorflow as tf
from model import SwinTransformer
from data import train_dataset, val_dataset, infer_dataset
from mobilenet_v2 import mobilenet_v2
import pickle as pkl
import os
import numpy as np

flags = tf.app.flags
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 2, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_integer('val_batch_size', 128, 'batch size')
flags.DEFINE_integer('num_classes', 10, 'num classes')
flags.DEFINE_string('model_path', './swin_base_224', 'path to model ckpt')
flags.DEFINE_string('model_name', 'swin_base_224', 'model name')
flags.DEFINE_string('mode', 'train', 'train/evaluate/inference')
flags.DEFINE_string('train_data_dir', './train_data', 'Directory to put the training data.')
flags.DEFINE_string('val_data_dir', './val_data', 'Directory to put the validation data.')
flags.DEFINE_string('infer_data_dir', './infer_data', 'Directory to put the inference data.')
flags.DEFINE_string('output', './output', 'Directory to save model.')
flags.DEFINE_string('label_to_index', './label_to_index.pkl', 'Directory to save model.')
flags.DEFINE_string('model_choice', 'swin', 'swin/mobile')
FLAGS = flags.FLAGS

IMAGE_SIZE = {
    "swin": 224,
    "mobile": 224
}


def get_model(model_path, model_name=""):
    if FLAGS.model_choice == "swin":
        return [SwinTransformer(model_path, model_name, include_top=False, pretrained=True)]
    else:
        return [mobilenet_v2(model_path, include_top=False),
                tf.keras.layers.GlobalAveragePooling2D()]


def main(_):
    if FLAGS.mode == "train":
        model = tf.keras.Sequential([
            tf.keras.layers.Lambda(
                lambda data: tf.keras.applications.imagenet_utils.preprocess_input(tf.cast(data, tf.float32),
                                                                                   mode="torch"),
                input_shape=[IMAGE_SIZE[FLAGS.model_choice], IMAGE_SIZE[FLAGS.model_choice], 3]),
            *get_model(FLAGS.model_path, FLAGS.model_name),
            tf.keras.layers.Dense(FLAGS.num_classes, activation='softmax')
        ])
        checkpoint_path = "%s/{epoch:04d}/%s.ckpt" % (FLAGS.output, FLAGS.model_name)

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                save_weights_only=True,
                verbose=1,
                save_freq='epoch'),
        ]

        samples_num, label_to_index, train_ds = train_dataset(FLAGS.train_data_dir,
                                                              IMAGE_SIZE[FLAGS.model_choice],
                                                              batch_size=FLAGS.batch_size)
        _, _, val_ds = val_dataset(FLAGS.val_data_dir,
                                   IMAGE_SIZE[FLAGS.model_choice],
                                   label_to_index,
                                   batch_size=FLAGS.val_batch_size)

        pkl.dump(label_to_index, open(os.path.join(FLAGS.output, "label_to_index.pkl"), "wb"))

        steps_per_epoch = samples_num // FLAGS.batch_size

        model = tf.keras.utils.multi_gpu_model(model, gpus=2)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(FLAGS.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=["accuracy"]
        )

        history = model.fit(train_ds, epochs=FLAGS.epochs,
                            validation_data=val_ds, callbacks=callbacks,
                            steps_per_epoch=steps_per_epoch
                            )
        pkl.dump(history.history, open(os.path.join(FLAGS.output, "history"), "wb"))

        if FLAGS.mode == "eval":
            model = tf.keras.Sequential([
                tf.keras.layers.Lambda(
                    lambda data: tf.keras.applications.imagenet_utils.preprocess_input(tf.cast(data, tf.float32),
                                                                                       mode="torch"),
                    input_shape=[IMAGE_SIZE[FLAGS.model_choice], IMAGE_SIZE[FLAGS.model_choice], 3]),
                *get_model(FLAGS.model_choice, FLAGS.model_name),
                tf.keras.layers.Dense(FLAGS.num_classes, activation='softmax')
            ])
            label_to_index = pkl.load(open(FLAGS.label_to_index, "rb"))
            samples_num, _, val_ds = val_dataset(FLAGS.val_data_dir, IMAGE_SIZE[FLAGS.model_choice], label_to_index,
                                                 batch_size=FLAGS.val_batch_size)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(FLAGS.learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=["accuracy"]
            )
            print("Evaluate on test data")
            results = model.evaluate(val_ds)
            print("test loss, test acc:", results)

        if FLAGS.mode == "infer":
            model = tf.keras.Sequential([
                tf.keras.layers.Lambda(
                    lambda data: tf.keras.applications.imagenet_utils.preprocess_input(tf.cast(data, tf.float32),
                                                                                       mode="torch"),
                    input_shape=[IMAGE_SIZE[FLAGS.model_choice], IMAGE_SIZE[FLAGS.model_choice], 3]),
                *get_model(FLAGS.model_choice, FLAGS.model_name),
                tf.keras.layers.Dense(FLAGS.num_classes, activation='softmax')
            ])
            _, _, infer_ds = infer_dataset(FLAGS.infer_data_dir, IMAGE_SIZE[FLAGS.model_choice],
                                           batch_size=FLAGS.val_batch_size)

            inference_results = model.predict(infer_ds)
            print("Finish inference:", inference_results.shape)
            np.save(os.path.join(FLAGS.output, "inference_results.npy"), inference_results)


if __name__ == "__main__":
    main(0)
