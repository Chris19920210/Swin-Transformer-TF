import tensorflow as tf
from modelzoo.swin_model import SwinTransformer
from data_processing.data import train_dataset, val_dataset, infer_dataset
from modelzoo.mobilenet_v2 import mobilenet_v2
import pickle as pkl
import os
import numpy as np
from utils import top3_acc, top5_acc, WarmUpCosineDecayScheduler, get_lr_metric, EvalPerClass
from SparseCategoricalFocalLoss import SparseCategoricalFocalLoss
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from utils import get_multi_tasks_model

flags = tf.compat.v1.flags
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 2, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_integer('val_batch_size', 128, 'batch size')
flags.DEFINE_integer('num_classes_task0', 10, 'num classes for task0')
flags.DEFINE_integer('num_classes_task1', 10, 'num classes for task1')
flags.DEFINE_string('model_path', './swin_base_224', 'path to model ckpt')
flags.DEFINE_string('model_name', 'swin_base_224', 'model name')
flags.DEFINE_string('mode', 'train', 'train/evaluate/inference')
flags.DEFINE_string('train_data_dir', './train_data', 'Directory to put the training data.')
flags.DEFINE_string('val_data_dir', './val_data', 'Directory to put the validation data.')
flags.DEFINE_string('infer_data_dir', './infer_data', 'Directory to put the inference data.')
flags.DEFINE_string('output', './output', 'Directory to save model.')
flags.DEFINE_string('task_to_task0', './task_to_task0.pkl', 'Directory to task_to_task0.')
flags.DEFINE_string('label_to_index_task0', './label_to_index_task0.pkl', 'Directory to label_to_index_task0.')
flags.DEFINE_string('task_to_task1', './task_to_task1.pkl', 'Directory to task_to_task1.')
flags.DEFINE_string('label_to_index_task1', './label_to_index_task1.pkl', 'Directory to label_to_index_task1.')
flags.DEFINE_integer('warmup_epochs', 10, 'Directory to save model.')
flags.DEFINE_string('model_choice', 'swin', 'swin/mobile')
flags.DEFINE_integer('gpus', 1, 'nums of gpus')
flags.DEFINE_boolean('eval_per_class', True, 'whether evaluate per class')
flags.DEFINE_string('mapping', None, 'path to mapping')
flags.DEFINE_boolean('focal', True, 'whether use focal loss')
flags.DEFINE_boolean('with_probs', True, 'whether save probs')
flags.DEFINE_float('dropout', 0., "dropout rate")
flags.DEFINE_list('weights_for_classes', [1.0, 1.0], 'weights for loss')
flags.DEFINE_list('top_layer_sizes', [128, 64], 'top layer sizes')
FLAGS = flags.FLAGS

IMAGE_SIZE = {
    "swin": 224,
    "mobile": 224
}


def get_model(model_path, num_classes_task0, num_classes_task1, model_name="", pretrained=True, is_training=False):
    if FLAGS.model_choice == "swin":
        if not is_training:
            model = SwinTransformer(model_path, model_name, include_top=False, pretrained=pretrained)
        else:
            model = SwinTransformer(model_path, model_name, include_top=False, drop_rate=FLAGS.dropout,
                                    attn_drop_rate=FLAGS.dropout, drop_path_rate=FLAGS.dropout,
                                    pretrained=pretrained)
    else:
        model = mobilenet_v2(model_path, include_top=False)

    model = get_multi_tasks_model(model, num_classes_task0, num_classes_task1, list(map(int, FLAGS.top_layer_sizes)))

    return model


def main(_):
    if FLAGS.mode == "train":
        device_list = ["/gpu:%d" % i for i in range(FLAGS.gpus)]
        strategy = tf.distribute.MirroredStrategy(devices=device_list)
        with strategy.scope():
            model = get_model(FLAGS.model_path, FLAGS.num_classes_task0, FLAGS.num_classes_task1, FLAGS.model_name,
                              is_training=True)
            optimizer = tf.keras.optimizers.Adam()
            lr_metric = get_lr_metric(optimizer)
            model.compile(
                optimizer=optimizer,
                loss=[SparseCategoricalFocalLoss(gamma=2) if FLAGS.focal else SparseCategoricalCrossentropy(),
                      SparseCategoricalFocalLoss(gamma=2) if FLAGS.focal else SparseCategoricalCrossentropy()],
                loss_weights=list(map(float, FLAGS.weights_for_classes)),
                metrics=[["sparse_categorical_accuracy", top3_acc, top5_acc, lr_metric],
                         ["sparse_categorical_accuracy", top3_acc, top5_acc, lr_metric]]
            )
            checkpoint_path = "%s/{epoch:04d}/%s.ckpt" % (FLAGS.output, FLAGS.model_name)

            # load data
            batch_size = FLAGS.batch_size * strategy.num_replicas_in_sync
            val_batch_size = FLAGS.val_batch_size * strategy.num_replicas_in_sync
            samples_num, task_to_task0, label_to_index_task0, task_to_task1, label_to_index_task1, train_ds = \
                train_dataset(
                    FLAGS.train_data_dir,
                    IMAGE_SIZE[FLAGS.model_choice],
                    batch_size=batch_size)

            _, _, _, _, _, val_ds = val_dataset(FLAGS.val_data_dir,
                                                IMAGE_SIZE[FLAGS.model_choice],
                                                task_to_task0,
                                                label_to_index_task0,
                                                task_to_task1,
                                                label_to_index_task1,
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

            pkl.dump(task_to_task0, open(os.path.join(FLAGS.output, "task_to_task0.pkl"), "wb"))
            pkl.dump(label_to_index_task0, open(os.path.join(FLAGS.output, "label_to_index_task0.pkl"), "wb"))
            pkl.dump(task_to_task1, open(os.path.join(FLAGS.output, "task_to_task1.pkl"), "wb"))
            pkl.dump(label_to_index_task1, open(os.path.join(FLAGS.output, "label_to_index_task1.pkl"), "wb"))

            steps_per_epoch = samples_num // batch_size

            history = model.fit(train_ds, epochs=FLAGS.epochs,
                                validation_data=val_ds, callbacks=callbacks,
                                steps_per_epoch=steps_per_epoch
                                )
            pkl.dump(history.history, open(os.path.join(FLAGS.output, "history"), "wb"))

    if FLAGS.mode == "eval":
        model = get_model(FLAGS.model_path, FLAGS.num_classes_task0, FLAGS.num_classes_task1, FLAGS.model_name,
                          pretrained=False, is_training=False)
        model.load_weights(FLAGS.model_path)
        tf.keras.backend.set_learning_phase(0)
        task_to_task0 = pkl.load(open(FLAGS.task_to_task0, "rb"))
        label_to_index_task0 = pkl.load(open(FLAGS.label_to_index_task0, "rb"))
        task_to_task1 = pkl.load(open(FLAGS.task_to_task1, "rb"))
        label_to_index_task1 = pkl.load(open(FLAGS.label_to_index_task1, "rb"))
        model.save(FLAGS.output, include_optimizer=False)
        print("Model save without optimizer, Done!")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(FLAGS.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=[["sparse_categorical_accuracy", top3_acc, top5_acc],
                     ["sparse_categorical_accuracy", top3_acc, top5_acc]]
        )
        print("Evaluate on test data")

        if FLAGS.eval_per_class:
            samples_num, _, _, _, _, val_ds_with_trace = val_dataset(FLAGS.val_data_dir,
                                                                     IMAGE_SIZE[FLAGS.model_choice],
                                                                     task_to_task0, label_to_index_task0,
                                                                     task_to_task1, label_to_index_task1,
                                                                     batch_size=FLAGS.val_batch_size, tracer=True)

            per_class_evaluator_task0 = EvalPerClass(label_to_index_task0)
            per_class_evaluator_task1 = EvalPerClass(label_to_index_task1)

            for i, (x_test, (y_test_task0, y_test_task1), path) in enumerate(val_ds_with_trace.as_numpy_iterator()):
                y_probs_task0, y_probs_task1 = model.predict(x_test)
                y_pred_task0, y_pred_task1 = np.argmax(y_probs_task0, axis=-1), np.argmax(y_probs_task1, axis=-1)

                if FLAGS.with_probs:
                    per_class_evaluator_task0(y_test_task0, y_pred_task0, path, y_probs_task0)
                    per_class_evaluator_task1(y_test_task1, y_pred_task1, path, y_probs_task1)
                if i % 10 == 0:
                    per_class_evaluator_task0.eval('Task 0 Eval after %d iter' % i)
                    per_class_evaluator_task1.eval('Task 1 Eval after %d iter' % i)
            per_class_evaluator_task0.eval('Task 0 Final')
            per_class_evaluator_task1.eval('Task 1 Final')
            per_class_evaluator_task0.save_trace(os.path.join(FLAGS.output, "task0_tracer.pkl"))
            per_class_evaluator_task1.save_trace(os.path.join(FLAGS.output, "task1_tracer.pkl"))
            if FLAGS.with_probs:
                per_class_evaluator_task0.save_prob_trace(os.path.join(FLAGS.output, "task0_prob_tracer.pkl"))
                per_class_evaluator_task1.save_prob_trace(os.path.join(FLAGS.output, "task1_prob_tracer.pkl"))

        samples_num, _, _, _, val_ds = val_dataset(FLAGS.val_data_dir, IMAGE_SIZE[FLAGS.model_choice],
                                                   task_to_task0, label_to_index_task0,
                                                   task_to_task1, label_to_index_task1,
                                                   batch_size=FLAGS.val_batch_size)
        results = model.evaluate(val_ds)
        print("test loss, test acc:", results)

    if FLAGS.mode == "infer":
        model = tf.keras.models.load_model(FLAGS.model_path)
        _, _, _, _, infer_ds = infer_dataset(FLAGS.infer_data_dir, IMAGE_SIZE[FLAGS.model_choice],
                                             batch_size=FLAGS.val_batch_size)

        task1_inference_results, task2_inference_results = model.predict(infer_ds)
        print("Task1 Finish inference:", task1_inference_results.shape)
        print("Task2 Finish inference:", task2_inference_results.shape)
        np.save(os.path.join(FLAGS.output, "task1_inference_results.npy"), task1_inference_results)
        np.save(os.path.join(FLAGS.output, "task2_inference_results.npy"), task2_inference_results)


if __name__ == "__main__":
    main(0)
