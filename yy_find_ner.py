# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf
import numpy as np
from segment import SegmentClass
import jieba
import lstm_crf_layer
from lstm_crf_layer import BLSTM_CRF
from tensorflow_core.contrib.layers.python.layers import initializers
import codecs
import tensorflow.metrics
import tf_metrics
from numpy import ndarray

flags = tf.flags

FLAGS = flags.FLAGS

# 输入的语料库文件
CORPUS_DIR = "extractor_output_data_4"

# CORPUS_DIR = 'baidu'

## Required parameters
flags.DEFINE_string(
    "data_dir", "test/train_data/" + CORPUS_DIR,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", "test/chinese_wwm_ext_L-12_H-768_A-12/bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", "yy", "The name of the task to train.")

flags.DEFINE_string("vocab_file", "test/chinese_wwm_ext_L-12_H-768_A-12/vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", "test/output/" + CORPUS_DIR,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", "test/chinese_wwm_ext_L-12_H-768_A-12/bert_model.ckpt",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_export", False, "Whether to export the model.")
flags.DEFINE_string("export_dir", "test/output/" + CORPUS_DIR + "/bert_model",
                    "The dir where the exported model will be written.")

flags.DEFINE_bool(
    "do_predict", True,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_bool("do_input_test", True, "Whether to run predict input.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 2e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 2.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

# lstm parame
flags.DEFINE_integer('lstm_size', 128, 'size of lstm units')
flags.DEFINE_integer('num_layers', 1, 'number of rnn layers, default is 1')
flags.DEFINE_string('cell', 'lstm', 'which rnn cell used')
flags.DEFINE_float('droupout_rate', 0.5, 'Dropout rate')


# 单个输入数据的包装类
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, label_list=None):
        """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.label_list = label_list
        self.token_list = None

        def set_token_list(self, tokens):
            self.token_list = tokens


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_ids,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class YYDataProcessor(DataProcessor):
    """ 使用https://github.com/dbiir/UER-py链接中提供的Datasets LCQMC"""

    def _createexamples(self, data_path, max_count=100, mode='test'):
        # with open(file_path, 'r', encoding="utf-8") as f:
        #     reader = f.readlines()
        with open(data_path, 'r', encoding="utf-8") as f:
            input_csv_reader = csv.reader(f)
            reader = list(input_csv_reader)
        examples = []
        for index, line in enumerate(reader):
            if index == 0:
                continue
            if max_count > 0:
                if index > max_count:
                    break
            guid = mode + ('-%d' % index)
            # split_line = line.strip().split("\t")
            # print(split_line)
            split_line = line
            if (len(split_line[0].strip()) == 0):
                continue
            text_a = tokenization.convert_to_unicode(split_line[0])
            if len(split_line) > 1:
                labelstring = tokenization.convert_to_unicode(split_line[1])
                label_list = labelstring.split('_')
                label_list = label_list[0: len(label_list) - 1]
            else:
                label_list = "0" * len(text_a)
            examples.append(InputExample(guid, text_a, label_list))
        return examples

    def get_train_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'train.csv')
        print("----------读train数据-----------")
        examples = self._createexamples(file_path, 1600, 'train')
        print("----------读train数据 end-----------")
        return examples

    def get_dev_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'test.csv')
        print("----------读dev数据-----------")
        examples = self._createexamples(file_path, 0, 'dev')
        print("----------读dev数据 end-----------")
        return examples

    def get_test_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'test.csv')
        print("----------读test数据-----------")
        examples = self._createexamples(file_path, 0, 'test')
        print("----------读test数据 end-----------")
        return examples

    def get_labels(self):
        return ["0", "ner", "sub"]


"""
将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中
:param ex_index: index
:param example: 一个样本
:param label_list: 标签列表
:param max_seq_length:
:param tokenizer:
:return:
"""


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_ids=[0] * max_seq_length,
            is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)

    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]

    example.token_list = tokens_a
    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    label_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    label_ids.append(0)
    # token是会把一些两个字的词算成一个token的
    # 比如yy是一个token
    word_index = 0
    for ind, token in enumerate(tokens_a):
        tokens.append(token)
        segment_ids.append(0)
        if word_index >= len(example.label_list):
            print(tokens_a)
            print('stop')
        item = example.label_list[word_index]
        if token.find('[') >= 0:
            word_index += 1
        else:
            tmp = token.replace('#', '')
            word_index += len(tmp)
        if (len(item) > 1):
            item = item[0:3]
        label_ids.append(label_map[item])
    tokens.append("[SEP]")
    segment_ids.append(0)
    label_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)
    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %s)" % (example.label_list, " ".join([str(x) for x in label_ids])))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        is_real_example=True)
    return feature


"""
    将数据转化为TF_Record 结构，作为模型数据输入
    :param examples:  样本
    :param label_list:标签list
    :param max_seq_length: 预先设定的最大序列长度
    :param tokenizer: tokenizer 对象
    :param output_file: tf.record 输出路径
    :return:
"""


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "is_real_example": tf.io.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    # 使用数据加载BertModel,获取对应的字embedding
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    embedding = model.get_sequence_output()
    max_seq_length = embedding.shape[1].value
    used = tf.sign(tf.abs(input_ids))
    lengths = tf.reduce_sum(used, reduction_indices=1)
    blstm_crf = BLSTM_CRF(embedded_chars=embedding, hidden_unit=FLAGS.lstm_size, cell_type=FLAGS.cell,
                          num_layers=FLAGS.num_layers,
                          dropout_rate=FLAGS.droupout_rate, initializers=initializers, num_labels=num_labels,
                          seq_length=max_seq_length, labels=labels, lengths=lengths, is_training=is_training)
    rst = blstm_crf.add_blstm_crf_layer(crf_only=True)
    return rst
    # 不支持test的
    # final_hidden = model.get_sequence_output()
    # final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
    # batch_size = final_hidden_shape[0]
    # seq_length = final_hidden_shape[1]
    # hidden_size = final_hidden_shape[2]
    #
    # output_weights = tf.get_variable(
    #     "output_weights", [num_labels, hidden_size],
    #     initializer=tf.truncated_normal_initializer(stddev=0.02))
    #
    # output_bias = tf.get_variable(
    #     "output_bias", [num_labels], initializer=tf.zeros_initializer())
    # with tf.variable_scope("loss"):
    #     final_hidden_matrix = tf.reshape(final_hidden, [batch_size * seq_length, hidden_size])
    #     logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
    #     logits = tf.nn.bias_add(logits, output_bias)
    #
    #     logits = tf.reshape(logits, [batch_size, seq_length, num_labels])
    #     # logits = tf.transpose(logits, [2,0,1])
    #     one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    #     log_probs = tf.nn.log_softmax(logits, axis=-1)
    #     # probabilities = tf.nn.softmax(logits, axis=-1)
    #     per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    #     loss = tf.reduce_mean(per_example_loss)
    #
    # return (loss, per_example_loss, logits )

    # 默认版本

    # output_layer = model.get_pooled_output()
    #
    # hidden_size = output_layer.shape[-1].value
    #
    # output_weights = tf.get_variable(
    #     "output_weights", [num_labels, hidden_size],
    #     initializer=tf.truncated_normal_initializer(stddev=0.02))
    #
    # output_bias = tf.get_variable(
    #     "output_bias", [num_labels], initializer=tf.zeros_initializer())
    #
    # with tf.variable_scope("loss"):
    #   if is_training:
    #     # I.e., 0.1 dropout
    #     output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
    #
    #   logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    #   logits = tf.nn.bias_add(logits, output_bias)
    #   probabilities = tf.nn.softmax(logits, axis=-1)
    #   log_probs = tf.nn.log_softmax(logits, axis=-1)
    #
    #   one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    #
    #   per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    #   loss = tf.reduce_mean(per_example_loss)
    #
    #   return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    # 构建模型
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # probabilities
        # 使用参数构建模型,input_idx 就是输入的样本idx表示，label_ids 就是标签的idx表示
        # (total_loss, per_example_loss, logits) = create_model(
        #     bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        #     num_labels, use_one_hot_embeddings)
        (total_loss, logits, trans, pred_ids) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.EVAL:

            # 针对NER ,进行了修改
            def metric_fn(label_ids, pred_ids):
                # 首先对结果进行维特比解码
                # crf 解码
                indices = [2, 3]  # indice参数告诉评估矩阵评估哪些标签,与label_list相对应
                weight = tf.sequence_mask(FLAGS.max_seq_length)
                precision = tf_metrics.precision(label_ids, pred_ids, num_labels, indices, weight)
                recall = tf_metrics.recall(label_ids, pred_ids, num_labels, indices, weight)
                f = tf_metrics.f1(label_ids, pred_ids, num_labels, indices, weight)

                return {
                    "eval_precision": precision,
                    "eval_recall": recall,
                    "eval_f": f,
                    # "eval_loss": loss,
                }

            eval_metrics = (metric_fn, [label_ids, pred_ids])
            # eval_metrics = (metric_fn, [label_ids, logits])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)

        else:
            # raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions=pred_ids,
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        features.append(feature)
    return features


# 模型保存
def serving_input_fn():
    label_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='label_ids')
    input_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_ids')
    input_mask = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_mask')
    segment_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='segment_ids')
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'label_ids': label_ids,
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids,
    })()
    return input_fn


def find_ner_in_single_sentence(estimator, label_list, tokenizer, question_string):
    exam = InputExample(str('input_test'), question_string, ['0'] * len(question_string))
    return find_ner_in_example(estimator, label_list, tokenizer, exam)


def find_ner_in_example(estimator, label_list, tokenizer, exam):
    feature = convert_single_example(10, exam, label_list, FLAGS.max_seq_length, tokenizer)
    prediction = estimator({
        "input_ids": [feature.input_ids],
        "input_mask": [feature.input_mask],
        "segment_ids": [feature.segment_ids],
        "label_ids": [feature.label_ids],
    })

    # predict_ids 是 ndarray
    predict_ids = prediction['output']
    # 需要去掉开头的 [CLS]对应的id
    real_ids = predict_ids.flatten().tolist()
    real_ids = real_ids[1: len(exam.token_list) + 1]
    ner_list = []
    sub_list = []
    tmp_ner = ''
    tmp_sub = ''
    for (index, predict_id) in enumerate(real_ids):
        if predict_id == 1:
            tmp_ner = tmp_ner + exam.token_list[index]
        elif len(tmp_ner) > 0:
            ner_list.append(tmp_ner)
            tmp_ner = ''
        elif predict_id == 2:
            tmp_sub = tmp_sub + exam.token_list[index]
        elif len(tmp_sub) > 0:
            sub_list.append(tmp_sub)
            tmp_sub = ''
    if len(tmp_ner) > 0:
        ner_list.append(tmp_ner)
    if len(tmp_sub) > 0:
        sub_list.append(tmp_sub)
    print(exam.text_a)
    print('找到目标：%s' % ner_list)
    print('找到子对象：%s' % sub_list)

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    predict_ids = predict_ids.flatten().tolist()
    for (index,value) in enumerate(feature.input_mask):
        if value == 1:
            if feature.label_ids[index] == predict_ids[index]:
                if predict_ids[index] == 0:
                    tn += 1
                else:
                    tp += 1
            else:
                if predict_ids[index] == 0:
                    fn += 1
                else:
                    fp += 1
    #去掉开始和结束的标记位
    tn -= 2;
    return [ner_list, sub_list, tp, tn, fp, fn]


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        "yy": YYDataProcessor,
    }

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)
    #
    # if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict and not FLAGS.do_input_test:
    #   raise ValueError(
    #       "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        print("开始训练")
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
        print("结束训练")
    if FLAGS.do_eval:
        print("开始评估")
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        num_actual_eval_examples = len(eval_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) % FLAGS.eval_batch_size != 0:
                eval_examples.append(PaddingInputExample())
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            assert len(eval_examples) % FLAGS.eval_batch_size == 0
            eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with codecs.open(output_eval_file, "w", encoding='utf-8') as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
        print("结束评估")

    if FLAGS.do_predict:
        estimator = tf.contrib.predictor.from_saved_model('test/output/extractor_output_data_4/bert_model/1586930471')
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        output_test_path = os.path.join(FLAGS.output_dir, 'test_out.csv')
        test_output_file = open(output_test_path, 'w+', encoding="utf-8")
        test_csv_writer = csv.writer(test_output_file)
        total_tp = 0
        total_tn = 0
        total_fp = 0
        total_fn = 0
        for (ex_index, example) in enumerate(predict_examples):
            (ner_list, sub_list, tp, tn, fp, fn) = find_ner_in_example(estimator, label_list, tokenizer, example)
            ner_string = '|'.join(ner_list)
            sub_string = '|'.join(sub_list)
            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn
            test_csv_writer.writerow([example.text_a, ner_string, sub_string])

        # precision
        precision = total_tp / (total_tp + total_fp + 0.0);
        recall = total_tp / (total_tp + total_fn)
        f_meature = 2.0 * precision * recall / (precision + recall)
        print("precision: %s\nrecall:%s\nf_meature:%s\n" % (str(precision), str(recall), str(f_meature)))
        print("结束测试")

    if FLAGS.do_input_test:
        input_index = 0
        user_dict_f = open('user_seg_dict.txt', 'r')
        jieba.load_userdict(user_dict_f)
        while True:
            input_string = input("请输入要测试的句子：")
            input_string = input_string.strip()
            if len(input_string) > 0:
                find_ner_in_single_sentence(estimator, label_list, tokenizer, input_string)
            input_index += 1

    if FLAGS.do_export and FLAGS.do_train:
        estimator._export_to_tpu = False
        estimator.export_savedmodel(FLAGS.export_dir, serving_input_fn)
    print("Done!")


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
