import tensorflow as tf
from modelzoo.swin_model import SwinTransformer
from data_processing.data import train_dataset, val_dataset, infer_dataset
from modelzoo.mobilenet_v2 import mobilenet_v2
import pickle as pkl
import os
import numpy as np
from utils import top3_acc, top5_acc, WarmUpCosineDecayScheduler, get_lr_metric

flags = tf.compat.v1.flags
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
flags.DEFINE_integer('warmup_epochs', 10, 'Directory to save model.')
flags.DEFINE_string('model_choice', 'swin', 'swin/mobile')
flags.DEFINE_integer('gpus', 1, 'nums of gpus')
FLAGS = flags.FLAGS

IMAGE_SIZE = {
    "swin": 224,
    "mobile": 224
}


def get_model(model_path, model_name="", pretrained=True):
    if FLAGS.model_choice == "swin":
        return [SwinTransformer(model_path, model_name, include_top=False, pretrained=pretrained)]
    else:
        return [mobilenet_v2(model_path, include_top=False),
                tf.keras.layers.GlobalAveragePooling2D()]


def main(_):
    if FLAGS.mode == "train":
        device_list = ["/gpu:%d" % i for i in range(FLAGS.gpus)]
        strategy = tf.distribute.MirroredStrategy(devices=device_list)
        with strategy.scope():
            model = tf.keras.Sequential([
                tf.keras.layers.Lambda(
                    lambda data: tf.keras.applications.imagenet_utils.preprocess_input(tf.cast(data, tf.float32),
                                                                                       mode="torch"),
                    input_shape=[IMAGE_SIZE[FLAGS.model_choice], IMAGE_SIZE[FLAGS.model_choice], 3]),
                *get_model(FLAGS.model_path, FLAGS.model_name),
                tf.keras.layers.Dense(FLAGS.num_classes, activation='softmax')
            ])
            optimizer = tf.keras.optimizers.Adam()
            lr_metric = get_lr_metric(optimizer)
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=["sparse_categorical_accuracy", top3_acc, top5_acc, lr_metric]
            )
        checkpoint_path = "%s/{epoch:04d}/%s.ckpt" % (FLAGS.output, FLAGS.model_name)

        # load data
        batch_size = FLAGS.batch_size * strategy.num_replicas_in_sync
        val_batch_size = FLAGS.val_batch_size * strategy.num_replicas_in_sync
        samples_num, label_to_index, train_ds = train_dataset(FLAGS.train_data_dir,
                                                              IMAGE_SIZE[FLAGS.model_choice],
                                                              batch_size=batch_size)

        _, _, val_ds = val_dataset(FLAGS.val_data_dir,
                                   IMAGE_SIZE[FLAGS.model_choice],
                                   label_to_index,
                                   batch_size=val_batch_size)

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

        train_ds = train_ds.with_options(options)
        val_ds = val_ds.with_options(options)

        # lr schedule
        total_steps = int(FLAGS.epochs * samples_num / batch_size)

        warmup_steps = int(FLAGS.warmup_epochs * samples_num / batch_size)

        warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=FLAGS.learning_rate,
                                                total_steps=total_steps,
                                                warmup_learning_rate=1e-6,
                                                warmup_steps=warmup_steps,
                                                hold_base_rate_steps=0)
        # callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                save_weights_only=True,
                verbose=1,
                save_freq='epoch'),
            warm_up_lr
        ]

        pkl.dump(label_to_index, open(os.path.join(FLAGS.output, "label_to_index.pkl"), "wb"))

        steps_per_epoch = samples_num // FLAGS.batch_size

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
            *get_model(FLAGS.model_path, FLAGS.model_name, False),
            tf.keras.layers.Dense(FLAGS.num_classes, activation='softmax')
        ])
        model.load_weights(FLAGS.model_path)
        label_to_index = pkl.load(open(FLAGS.label_to_index, "rb"))
        samples_num, _, val_ds = val_dataset(FLAGS.val_data_dir, IMAGE_SIZE[FLAGS.model_choice], label_to_index,
                                             batch_size=FLAGS.val_batch_size)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(FLAGS.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=["sparse_categorical_accuracy", top3_acc, top5_acc]
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
            *get_model(FLAGS.model_path, FLAGS.model_name, False),
            tf.keras.layers.Dense(FLAGS.num_classes, activation='softmax')
        ])
        model.load_weights(FLAGS.model_path)
        _, _, infer_ds = infer_dataset(FLAGS.infer_data_dir, IMAGE_SIZE[FLAGS.model_choice],
                                       batch_size=FLAGS.val_batch_size)

        inference_results = model.predict(infer_ds)
        print("Finish inference:", inference_results.shape)
        np.save(os.path.join(FLAGS.output, "inference_results.npy"), inference_results)


if __name__ == "__main__":
    main(0)
