#!/usr/local/bin/python
import os
import tensorflow as tf
import metrics as metrics
import stat_logger as stat_logger
from DataPipe import DataPipe
from ConfigLoader import logger


class Executor:

    def __init__(self, model, silence_step=200, skip_step=20):
        self.model = model
        self.silence_step = silence_step
        self.skip_step = skip_step
        self.pipe = DataPipe()

        self.saver = tf.train.Saver()
        self.tf_config = tf.ConfigProto(allow_soft_placement=True)
        self.tf_config.gpu_options.allow_growth = True

    def unit_test_train(self):
        with tf.Session() as sess:
            word_table_init = self.pipe.init_word_table()
            feed_table_init = {self.model.word_table_init: word_table_init}
            sess.run(tf.global_variables_initializer(), feed_dict=feed_table_init)
            logger.info('Word table init: done!')

            logger.info('Model: {0}, start a new session!'.format(self.model.model_name))

            n_iter = self.model.global_step.eval()

            # forward
            train_batch_loss_list = list()
            train_epoch_size = 0.0
            train_epoch_n_acc = 0.0
            train_batch_gen = self.pipe.batch_gen(phase='train')
            train_batch_dict = next(train_batch_gen)

            while n_iter < 100:
                feed_dict = {self.model.is_training_phase: True,
                             self.model.batch_size: train_batch_dict['batch_size'],
                             self.model.stock_ph: train_batch_dict['stock_batch'],
                             self.model.T_ph: train_batch_dict['T_batch'],
                             self.model.n_words_ph: train_batch_dict['n_words_batch'],
                             self.model.n_msgs_ph: train_batch_dict['n_msgs_batch'],
                             self.model.y_ph: train_batch_dict['y_batch'],
                             self.model.price_ph: train_batch_dict['price_batch'],
                             self.model.mv_percent_ph: train_batch_dict['mv_percent_batch'],
                             self.model.word_ph: train_batch_dict['word_batch'],
                             self.model.ss_index_ph: train_batch_dict['ss_index_batch'],
                             }

                ops = [self.model.y_T, self.model.y_T_, self.model.loss, self.model.optimize]
                train_batch_y, train_batch_y_, train_batch_loss, _ = sess.run(ops, feed_dict)

                # training batch stat
                train_epoch_size += float(train_batch_dict['batch_size'])
                train_batch_loss_list.append(train_batch_loss)
                train_batch_n_acc = sess.run(metrics.n_accurate(y=train_batch_y, y_=train_batch_y_))
                train_epoch_n_acc += float(train_batch_n_acc)

                stat_logger.print_batch_stat(n_iter, train_batch_loss, train_batch_n_acc,
                                             train_batch_dict['batch_size'])
                n_iter += 1

    def generation(self, sess, phase):
        generation_gen = self.pipe.batch_gen_by_stocks(phase)

        gen_loss_list = list()
        gen_size, gen_n_acc = 0.0, 0.0
        y_list, y_list_ = list(), list()

        for gen_batch_dict in generation_gen:

            feed_dict = {self.model.is_training_phase: False,
                         self.model.batch_size: gen_batch_dict['batch_size'],
                         self.model.stock_ph: gen_batch_dict['stock_batch'],
                         self.model.T_ph: gen_batch_dict['T_batch'],
                         self.model.n_words_ph: gen_batch_dict['n_words_batch'],
                         self.model.n_msgs_ph: gen_batch_dict['n_msgs_batch'],
                         self.model.y_ph: gen_batch_dict['y_batch'],
                         self.model.price_ph: gen_batch_dict['price_batch'],
                         self.model.mv_percent_ph: gen_batch_dict['mv_percent_batch'],
                         self.model.word_ph: gen_batch_dict['word_batch'],
                         self.model.ss_index_ph: gen_batch_dict['ss_index_batch'],
                         self.model.dropout_mel_in: 0.0,
                         self.model.dropout_mel: 0.0,
                         self.model.dropout_ce: 0.0,
                         self.model.dropout_vmd_in: 0.0,
                         self.model.dropout_vmd: 0.0,
                         }

            gen_batch_y, gen_batch_y_, gen_batch_loss = sess.run([self.model.y_T, self.model.y_T_, self.model.loss],
                                                                 feed_dict=feed_dict)

            # gather
            y_list.append(gen_batch_y)
            y_list_.append(gen_batch_y_)
            gen_loss_list.append(gen_batch_loss)  # list of floats

            gen_batch_n_acc = float(sess.run(metrics.n_accurate(y=gen_batch_y, y_=gen_batch_y_)))  # float
            gen_n_acc += gen_batch_n_acc

            batch_size = float(gen_batch_dict['batch_size'])
            gen_size += batch_size

        results = metrics.eval_res(gen_n_acc, gen_size, gen_loss_list, y_list, y_list_)
        return results

    def train_and_dev(self):
        with tf.Session(config=self.tf_config) as sess:
            # prep: writer and init
            writer = tf.summary.FileWriter(self.model.tf_graph_path, sess.graph)

            # init all vars with tables
            feed_table_init = {self.model.word_table_init: self.pipe.init_word_table()}
            sess.run(tf.global_variables_initializer(), feed_dict=feed_table_init)
            logger.info('Word table init: done!')

            # prep: checkpoint
            checkpoint = tf.train.get_checkpoint_state(os.path.dirname(self.model.tf_checkpoint_file_path))
            if checkpoint and checkpoint.model_checkpoint_path:
                # restore partial saved vars
                reader = tf.train.NewCheckpointReader(checkpoint.model_checkpoint_path)
                restore_dict = dict()
                for v in tf.all_variables():
                    tensor_name = v.name.split(':')[0]
                    if reader.has_tensor(tensor_name):
                        print('has tensor: {0}'.format(tensor_name))
                        restore_dict[tensor_name] = v

                checkpoint_saver = tf.train.Saver(restore_dict)
                checkpoint_saver.restore(sess, checkpoint.model_checkpoint_path)
                logger.info('Model: {0}, session restored!'.format(self.model.model_name))
            else:
                logger.info('Model: {0}, start a new session!'.format(self.model.model_name))

            for epoch in range(self.model.n_epochs):
                logger.info('Epoch: {0}/{1} start'.format(epoch+1, self.model.n_epochs))

                # training phase
                train_batch_loss_list = list()
                epoch_size, epoch_n_acc = 0.0, 0.0

                train_batch_gen = self.pipe.batch_gen(phase='train')  # a new gen for a new epoch

                for train_batch_dict in train_batch_gen:

                    # logger.info('train: batch_size: {0}'.format(train_batch_dict['batch_size']))

                    feed_dict = {self.model.is_training_phase: True,
                                 self.model.batch_size: train_batch_dict['batch_size'],
                                 self.model.stock_ph: train_batch_dict['stock_batch'],
                                 self.model.T_ph: train_batch_dict['T_batch'],
                                 self.model.n_words_ph: train_batch_dict['n_words_batch'],
                                 self.model.n_msgs_ph: train_batch_dict['n_msgs_batch'],
                                 self.model.y_ph: train_batch_dict['y_batch'],
                                 self.model.price_ph: train_batch_dict['price_batch'],
                                 self.model.mv_percent_ph: train_batch_dict['mv_percent_batch'],
                                 self.model.word_ph: train_batch_dict['word_batch'],
                                 self.model.ss_index_ph: train_batch_dict['ss_index_batch'],
                                 }

                    ops = [self.model.y_T, self.model.y_T_, self.model.loss, self.model.optimize,
                           self.model.global_step]
                    train_batch_y, train_batch_y_, train_batch_loss, _, n_iter = sess.run(ops, feed_dict)

                    # training batch stat
                    epoch_size += float(train_batch_dict['batch_size'])
                    train_batch_loss_list.append(train_batch_loss)  # list of floats
                    train_batch_n_acc = sess.run(metrics.n_accurate(y=train_batch_y, y_=train_batch_y_))  # float
                    epoch_n_acc += float(train_batch_n_acc)

                    # save model and generation
                    if n_iter >= self.silence_step and n_iter % self.skip_step == 0:
                        stat_logger.print_batch_stat(n_iter, train_batch_loss, train_batch_n_acc,
                                                     train_batch_dict['batch_size'])
                        self.saver.save(sess, self.model.tf_saver_path, n_iter)
                        res = self.generation(sess, phase='dev')
                        stat_logger.print_eval_res(res)

                # print training epoch stat
                epoch_loss, epoch_acc = metrics.basic_train_stat(train_batch_loss_list, epoch_n_acc, epoch_size)
                stat_logger.print_epoch_stat(epoch_loss=epoch_loss, epoch_acc=epoch_acc)

        writer.close()

    def restore_and_test(self):
        with tf.Session(config=self.tf_config) as sess:
            checkpoint = tf.train.get_checkpoint_state(os.path.dirname(self.model.tf_checkpoint_file_path))
            if checkpoint and checkpoint.model_checkpoint_path:
                logger.info('Model: {0}, session restored!'.format(self.model.model_name))
                self.saver.restore(sess, checkpoint.model_checkpoint_path)
            else:
                logger.info('Model: {0}: NOT found!'.format(self.model.model_name))
                raise IOError

            res = self.generation(sess, phase='test')
            stat_logger.print_eval_res(res)
