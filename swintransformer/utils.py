from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
import pickle as pkl


def top3_acc(labels, logits):
    return keras.metrics.sparse_top_k_categorical_accuracy(y_true=labels, y_pred=logits, k=3)


def top5_acc(labels, logits):
    return keras.metrics.sparse_top_k_categorical_accuracy(y_true=labels, y_pred=logits, k=5)


def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
    """Cosine decay schedule with warm up period.
    Cosine annealing learning rate as described in:
      Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
      ICLR 2017. https://arxiv.org/abs/1608.03983
    In this schedule, the learning rate grows linearly from warmup_learning_rate
    to learning_rate_base for warmup_steps, then transitions to a cosine decay
    schedule.
    Arguments:
        global_step {int} -- global step.
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.
    Keyword Arguments:
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
    Returns:
      a float representing learning rate.
    Raises:
      ValueError: if warmup_learning_rate is larger than learning_rate_base,
        or if warmup_steps is larger than total_steps.
    """

    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
        np.pi *
        (global_step - warmup_steps - hold_base_rate_steps
         ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)
    return np.where(global_step > total_steps, 0.0, learning_rate)


class WarmUpCosineDecayScheduler(keras.callbacks.Callback):
    """Cosine decay with warmup learning rate scheduler
    """

    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 verbose=0):
        """Constructor for cosine decay with warmup learning rate scheduler.
    Arguments:
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.
    Keyword Arguments:
        global_step_init {int} -- initial global step, e.g. from previous checkpoint.
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
        verbose {int} -- 0: quiet, 1: update messages. (default: {0})
        """

        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.verbose = verbose
        self.learning_rates = []

    def on_train_batch_begin(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_train_batch_end(self, batch, logs=None):
        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.total_steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps)
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_step + 1, lr))


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr

    return lr


class EvalPerClass(object):
    def __init__(self, labels_to_index, mapping=None):
        self.labels = sorted(labels_to_index, key=lambda key: labels_to_index[key])
        self.mapping = mapping
        if mapping is not None:
            self.mapping_to_index = {k: i for i, k in enumerate(set(self.mapping.values()))}
            self.id_mapping_with_idx = [self.mapping_to_index[self.mapping[key]] for key in self.labels]
            self.labels = sorted(self.mapping_to_index, key=lambda key: self.mapping_to_index[key])

        self.sample_accu = np.zeros(len(self.labels))
        self.class_accu = np.zeros(len(self.labels))
        self.tracer_list = [[] for _ in range(len(self.labels))]
        self.prob_tracer_list = [[] for _ in range(len(self.labels))]

    def __call__(self, y_true, y_pred, paths=None, probs=None):
        if paths is None:
            for true, pred in zip(y_true, y_pred):
                if self.mapping is not None:
                    true = self.id_mapping_with_idx[true]
                    pred = self.id_mapping_with_idx[pred]
                self.acc(true, pred)
        else:
            if probs is None:
                for true, pred, path in zip(y_true, y_pred, paths):
                    if self.mapping is not None:
                        true = self.id_mapping_with_idx[true]
                        pred = self.id_mapping_with_idx[pred]
                    self.acc(true, pred, path)
            else:
                for true, pred, path, y_prob in zip(y_true, y_pred, paths, probs):
                    for true, pred, path in zip(y_true, y_pred, paths):
                        if self.mapping is not None:
                            true = self.id_mapping_with_idx[true]
                            pred = self.id_mapping_with_idx[pred]
                        self.acc(true, pred, path, y_prob)

    def acc(self, true, pred, path=None, y_prob=None):
        self.sample_accu[true] += 1
        if true == pred:
            self.class_accu[true] += 1
        if path is not None:
            self.tracer(true, pred, path)
        if y_prob is not None and path is not None:
            self.prob_tracer(path, true, y_prob)

    def eval(self, stage):
        acc_per_class = self.class_accu / (self.sample_accu + 1e-6)
        print(stage)
        print('In total Acc:%.4f, Total Sample num :%d' % (sum(self.class_accu) / (sum(self.sample_accu) + 1e-6),
                                                           int(sum(self.sample_accu))))
        for label, acc, cnt in zip(self.labels, acc_per_class, self.sample_accu):
            print("label:%s, acc:%.4f, sample_num:%d" % (label, acc, cnt))

    def tracer(self, true, pred, path):
        if true != pred:
            self.tracer_list[true].append(path.decode("utf-8"))

    def prob_tracer(self, path, true, y_prob):
        self.prob_tracer_list[true].append({'path': path, 'y_prob': y_prob})

    def save_trace(self, output_path):
        trace_result = {}
        for i, label in enumerate(self.labels):
            trace_result[label] = self.tracer_list[i]
        pkl.dump(trace_result, open(output_path, "wb"))

    def save_prob_tracer(self, output_path):
        prob_trace_result = {}
        for i, label in enumerate(self.labels):
            prob_trace_result[label] = self.prob_tracer_list[i]
        pkl.dump(prob_trace_result, open(output_path, "wb"))