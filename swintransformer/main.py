import tensorflow as tf
from model import SwinTransformer
from utils import DataAugmentation, train_dataset, val_dataset, infer_dataset
import pickle as pkl
import os
import numpy as np

flags = tf.app.flags
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 2, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_integer('num_classes', 10, 'num classes')
flags.DEFINE_string('model_path', './swin_base_224', 'path to model ckpt')
flags.DEFINE_string('model_name', 'swin_base_224', 'model name')
flags.DEFINE_string('mode', 'train', 'train/evaluate/inference')
flags.DEFINE_string('train_data_dir', './train_data', 'Directory to put the training data.')
flags.DEFINE_string('val_data_dir', './val_data', 'Directory to put the validation data.')
flags.DEFINE_string('infer_data_dir', './infer_data', 'Directory to put the inference data.')
flags.DEFINE_string('output', './output', 'Directory to save model.')
flags.DEFINE_string('label_to_index', './label_to_index.pkl', 'Directory to save model.')
FLAGS = flags.FLAGS

IMAGE_SIZE = (224, 224)


def main(_):
    if FLAGS.mode == "train":
        model = tf.keras.Sequential([
            tf.keras.layers.Lambda(
                lambda data: tf.keras.applications.imagenet_utils.preprocess_input(tf.cast(data, tf.float32), mode="torch"),
                input_shape=[*IMAGE_SIZE, 3]),
            SwinTransformer(FLAGS.model_path, FLAGS.model_name, include_top=False, pretrained=True),
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

        label_to_index, train_ds = train_dataset(FLAGS.train_data_dir, IMAGE_SIZE, batch_size=FLAGS.batch_size, epochs=FLAGS.epochs)
        _, val_ds = val_dataset(FLAGS.val_data_dir, IMAGE_SIZE, label_to_index, batch_size=FLAGS.batch_size, shuffle=False)
        pkl.dump(label_to_index, open(os.path.join(FLAGS.output, "label_to_index.pkl"), "wb"))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(FLAGS.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=["accuracy"]
        )

        history = model.fit(train_ds, epochs=FLAGS.epochs, validation_data=val_ds, callbacks=callbacks)
        pkl.dump(history.history, open(os.path.join(FLAGS.output, "history"), "wb"))


        if FLAGS.mode == "eval":
            model = tf.keras.Sequential([
                tf.keras.layers.Lambda(
                    lambda data: tf.keras.applications.imagenet_utils.preprocess_input(tf.cast(data, tf.float32),
                                                                                       mode="torch"),
                    input_shape=[*IMAGE_SIZE, 3]),
                SwinTransformer(FLAGS.model_path, FLAGS.model_name, include_top=False, pretrained=True),
                tf.keras.layers.Dense(FLAGS.num_classes, activation='softmax')
            ])
            label_to_index = pkl.load(open(FLAGS.label_to_index, "rb"))
            val_ds = val_dataset(FLAGS.val_data_dir, IMAGE_SIZE, label_to_index, batch_size=FLAGS.batch_size, shuffle=False)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(FLAGS.learning_rate),
                loss=tf.keras.metrics.SparseCategoricalAccuracy(),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
            )
            print("Evaluate on test data")
            results = model.evaluate(val_ds)
            print("test loss, test acc:", results)

        if FLAGS.mode == "infer":
            model = tf.keras.Sequential([
                tf.keras.layers.Lambda(
                    lambda data: tf.keras.applications.imagenet_utils.preprocess_input(tf.cast(data, tf.float32),
                                                                                       mode="torch"),
                    input_shape=[*IMAGE_SIZE, 3]),
                SwinTransformer(FLAGS.model_path, FLAGS.model_name, include_top=False, pretrained=True),
                tf.keras.layers.Dense(FLAGS.num_classes, activation='softmax')
            ])
            _, infer_ds = infer_dataset(FLAGS.infer_data_dir, IMAGE_SIZE, batch_size=FLAGS.batch_size, shuffle=False)

            inference_results = model.predict(infer_ds)
            print("Finish inference:", inference_results.shape)
            np.save(os.path.join(FLAGS.output, "inference_results.npy"), inference_results)


if __name__ == "__main__":
    main(0)
