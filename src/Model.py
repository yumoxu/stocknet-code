#!/usr/local/bin/python
from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
import neural as neural
import tensorflow.contrib.distributions as ds
from tensorflow.contrib.layers import batch_norm
from ConfigLoader import logger, ss_size, vocab_size, config_model, path_parser


class Model:

    def __init__(self):
        logger.info('INIT: #stock: {0}, #vocab+1: {1}'.format(ss_size, vocab_size))

        # model config
        self.mode = config_model['mode']
        self.opt = config_model['opt']
        self.lr = config_model['lr']
        self.decay_step = config_model['decay_step']
        self.decay_rate = config_model['decay_rate']
        self.momentum = config_model['momentum']

        self.kl_lambda_anneal_rate = config_model['kl_lambda_anneal_rate']
        self.kl_lambda_start_step = config_model['kl_lambda_start_step']
        self.use_constant_kl_lambda = config_model['use_constant_kl_lambda']
        self.constant_kl_lambda = config_model['constant_kl_lambda']

        self.daily_att = config_model['daily_att']
        self.alpha = config_model['alpha']

        self.clip = config_model['clip']
        self.n_epochs = config_model['n_epochs']
        self.batch_size_for_name = config_model['batch_size']

        self.max_n_days = config_model['max_n_days']
        self.max_n_msgs = config_model['max_n_msgs']
        self.max_n_words = config_model['max_n_words']

        self.weight_init = config_model['weight_init']
        uniform = True if self.weight_init == 'xavier-uniform' else False
        self.initializer = tf.contrib.layers.xavier_initializer(uniform=uniform)
        self.bias_initializer = tf.constant_initializer(0.0, dtype=tf.float32)

        self.word_embed_type = config_model['word_embed_type']

        self.y_size = config_model['y_size']
        self.word_embed_size = config_model['word_embed_size']
        self.stock_embed_size = config_model['stock_embed_size']
        self.price_embed_size = config_model['word_embed_size']

        self.mel_cell_type = config_model['mel_cell_type']
        self.variant_type = config_model['variant_type']
        self.vmd_cell_type = config_model['vmd_cell_type']

        self.vmd_rec = config_model['vmd_rec']

        self.mel_h_size = config_model['mel_h_size']
        self.msg_embed_size = config_model['mel_h_size']
        self.corpus_embed_size = config_model['mel_h_size']

        self.h_size = config_model['h_size']
        self.z_size = config_model['h_size']
        self.g_size = config_model['g_size']
        self.use_in_bn= config_model['use_in_bn']
        self.use_o_bn = config_model['use_o_bn']
        self.use_g_bn = config_model['use_g_bn']

        self.dropout_train_mel_in = config_model['dropout_mel_in']
        self.dropout_train_mel = config_model['dropout_mel']
        self.dropout_train_ce = config_model['dropout_ce']
        self.dropout_train_vmd_in = config_model['dropout_vmd_in']
        self.dropout_train_vmd = config_model['dropout_vmd']

        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        # model name
        name_pattern_max_n = 'days-{0}.msgs-{1}-words-{2}'
        name_max_n = name_pattern_max_n.format(self.max_n_days, self.max_n_msgs, self.max_n_words)

        name_pattern_input_type = 'word_embed-{0}.vmd_in-{1}'
        name_input_type = name_pattern_input_type.format(self.word_embed_type, self.variant_type)

        name_pattern_key = 'alpha-{0}.anneal-{1}.rec-{2}'
        name_key = name_pattern_key.format(self.alpha, self.kl_lambda_anneal_rate, self.vmd_rec)

        name_pattern_train = 'batch-{0}.opt-{1}.lr-{2}-drop-{3}-cell-{4}'
        name_train = name_pattern_train.format(self.batch_size_for_name, self.opt, self.lr, self.dropout_train_mel_in, self.mel_cell_type)

        name_tuple = (self.mode, name_max_n, name_input_type, name_key, name_train)
        self.model_name = '_'.join(name_tuple)

        # paths
        self.tf_graph_path = os.path.join(path_parser.graphs, self.model_name)  # summary
        self.tf_checkpoints_path = os.path.join(path_parser.checkpoints, self.model_name)  # checkpoints
        self.tf_checkpoint_file_path = os.path.join(self.tf_checkpoints_path, 'checkpoint')  # for restore
        self.tf_saver_path = os.path.join(self.tf_checkpoints_path, 'sess')  # for save

        # verification
        assert self.opt in ('sgd', 'adam')
        assert self.mel_cell_type in ('ln-lstm', 'gru', 'basic')
        assert self.vmd_cell_type in ('ln-lstm', 'gru')
        assert self.variant_type in ('hedge', 'fund', 'tech', 'discriminative')
        assert self.vmd_rec in ('zh', 'h')
        assert self.weight_init in ('xavier-uniform', 'xavier-normal')

    def _build_placeholders(self):
        with tf.name_scope('placeholder'):
            self.is_training_phase = tf.placeholder(dtype=tf.bool, shape=())
            self.batch_size = tf.placeholder(dtype=tf.int32, shape=())

            # init
            self.word_table_init = tf.placeholder(dtype=tf.float32, shape=[vocab_size, self.word_embed_size])

            # model
            self.stock_ph = tf.placeholder(dtype=tf.int32, shape=[None])
            self.T_ph = tf.placeholder(dtype=tf.int32, shape=[None, ])
            self.n_words_ph = tf.placeholder(dtype=tf.int32, shape=[None, self.max_n_days, self.max_n_msgs])
            self.n_msgs_ph = tf.placeholder(dtype=tf.int32, shape=[None, self.max_n_days])
            self.y_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.max_n_days, self.y_size])  # 2-d vectorised movement
            self.mv_percent_ph = tf.placeholder(dtype=tf.int32, shape=[None, self.max_n_days])  # movement percent
            self.price_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.max_n_days, 3])  # high, low, close
            self.word_ph = tf.placeholder(dtype=tf.int32, shape=[None, self.max_n_days, self.max_n_msgs, self.max_n_words])
            self.ss_index_ph = tf.placeholder(dtype=tf.int32, shape=[None, self.max_n_days, self.max_n_msgs])

            # dropout
            self.dropout_mel_in = tf.placeholder_with_default(self.dropout_train_mel_in, shape=())
            self.dropout_mel = tf.placeholder_with_default(self.dropout_train_mel, shape=())
            self.dropout_ce = tf.placeholder_with_default(self.dropout_train_ce, shape=())
            self.dropout_vmd_in = tf.placeholder_with_default(self.dropout_train_vmd_in, shape=())
            self.dropout_vmd = tf.placeholder_with_default(self.dropout_train_vmd, shape=())

    def _build_embeds(self):
        with tf.name_scope('embeds'):
            with tf.variable_scope('embeds'):
                word_table = tf.get_variable('word_table', initializer=self.word_table_init, trainable=False)
                self.word_embed = tf.nn.embedding_lookup(word_table, self.word_ph, name='word_embed')

    def _create_msg_embed_layer_in(self):
        """
            acquire the inputs for MEL.

            Input:
                word_embed: batch_size * max_n_days * max_n_msgs * max_n_words * word_embed_size

            Output:
                mel_in: same as word_embed
        """
        with tf.name_scope('mel_in'):
            with tf.variable_scope('mel_in'):
                mel_in = self.word_embed
                if self.use_in_bn:
                    mel_in = neural.bn(mel_in, self.is_training_phase, bn_scope='bn-mel_inputs')
                self.mel_in = tf.nn.dropout(mel_in, keep_prob=1-self.dropout_mel_in)

    def _create_msg_embed_layer(self):
        """
            Input:
                mel_in: same as word_embed

            Output:
                msg_embed: batch_size * max_n_days * max_n_msgs * msg_embed_size
        """

        def _for_one_trading_day(daily_in, daily_ss_index_vec, daily_mask):
            """
                daily_in: max_n_msgs * max_n_words * word_embed_size
            """
            out, _ = tf.nn.bidirectional_dynamic_rnn(mel_cell_f, mel_cell_b, daily_in, daily_mask,
                                                     mel_init_f, mel_init_b, dtype=tf.float32)
            out_f, out_b = out
            ss_indices = tf.reshape(daily_ss_index_vec, [-1, 1])

            msg_ids = tf.constant(range(0, self.max_n_msgs), dtype=tf.int32, shape=[self.max_n_msgs, 1])  # [0, 1, 2, ...]
            out_id = tf.concat([msg_ids, ss_indices], axis=1)
            # fw, bw and average
            mel_h_f, mel_h_b = tf.gather_nd(out_f, out_id), tf.gather_nd(out_b, out_id)
            msg_embed = (mel_h_f + mel_h_b) / 2

            return msg_embed

        def _for_one_sample(sample, sample_ss_index, sample_mask):
            return neural.iter(size=self.max_n_days, func=_for_one_trading_day,
                               iter_arg=sample, iter_arg2=sample_ss_index, iter_arg3=sample_mask)

        def _for_one_batch():
            return neural.iter(size=self.batch_size, func=_for_one_sample,
                               iter_arg=self.mel_in, iter_arg2=self.ss_index_ph, iter_arg3=self.n_words_ph)

        with tf.name_scope('mel'):
            with tf.variable_scope('mel_iter', reuse=tf.AUTO_REUSE):
                if self.mel_cell_type == 'ln-lstm':
                    mel_cell_f = tf.contrib.rnn.LayerNormBasicLSTMCell(self.mel_h_size)
                    mel_cell_b = tf.contrib.rnn.LayerNormBasicLSTMCell(self.mel_h_size)
                elif self.mel_cell_type == 'gru':
                    mel_cell_f = tf.contrib.rnn.GRUCell(self.mel_h_size)
                    mel_cell_b = tf.contrib.rnn.GRUCell(self.mel_h_size)
                else:
                    mel_cell_f = tf.contrib.rnn.BasicRNNCell(self.mel_h_size)
                    mel_cell_b = tf.contrib.rnn.BasicRNNCell(self.mel_h_size)

                mel_cell_f = tf.contrib.rnn.DropoutWrapper(mel_cell_f, output_keep_prob=1.0-self.dropout_mel)
                mel_cell_b = tf.contrib.rnn.DropoutWrapper(mel_cell_b, output_keep_prob=1.0-self.dropout_mel)

                mel_init_f = mel_cell_f.zero_state([self.max_n_msgs], tf.float32)
                mel_init_b = mel_cell_f.zero_state([self.max_n_msgs], tf.float32)

                msg_embed_shape = (self.batch_size, self.max_n_days, self.max_n_msgs, self.msg_embed_size)
                msg_embed = tf.reshape(_for_one_batch(), shape=msg_embed_shape)
                self.msg_embed = tf.nn.dropout(msg_embed, keep_prob=1-self.dropout_mel, name='msg_embed')

    def _create_corpus_embed(self):
        """
            msg_embed: batch_size * max_n_days * max_n_msgs * msg_embed_size

            => corpus_embed: batch_size * max_n_days * corpus_embed_size
        """
        with tf.name_scope('corpus_embed'):
            with tf.variable_scope('u_t'):
                proj_u = self._linear(self.msg_embed, self.msg_embed_size, 'tanh', use_bias=False)
                w_u = tf.get_variable('w_u', shape=(self.msg_embed_size, 1), initializer=self.initializer)
            u = tf.reduce_mean(tf.tensordot(proj_u, w_u, axes=1), axis=-1)  # batch_size * max_n_days * max_n_msgs

            mask_msgs = tf.sequence_mask(self.n_msgs_ph, maxlen=self.max_n_msgs, dtype=tf.bool, name='mask_msgs')
            ninf = tf.fill(tf.shape(mask_msgs), np.NINF)
            masked_score = tf.where(mask_msgs, u, ninf)
            u = neural.softmax(masked_score)  # batch_size * max_n_days * max_n_msgs
            u = tf.where(tf.is_nan(u), tf.zeros_like(u), u)  # replace nan with 0.0

            u = tf.expand_dims(u, axis=-2)  # batch_size * max_n_days * 1 * max_n_msgs
            corpus_embed = tf.matmul(u, self.msg_embed)  # batch_size * max_n_days * 1 * msg_embed_size
            corpus_embed = tf.reduce_mean(corpus_embed, axis=-2)  # batch_size * max_n_days * msg_embed_size
            self.corpus_embed = tf.nn.dropout(corpus_embed, keep_prob=1-self.dropout_ce, name='corpus_embed')

    def _build_mie(self):
        """
            Create market information encoder.

            corpus_embed: batch_size * max_n_days * corpus_embed_size
            price: batch_size * max_n_days * 3
            => x: batch_size * max_n_days * x_size
        """
        with tf.name_scope('mie'):
            self.price = self.price_ph
            self.price_size = 3

            if self.variant_type == 'tech':
                self.x = self.price
                self.x_size = self.price_size
            else:
                self._create_msg_embed_layer_in()
                self._create_msg_embed_layer()
                self._create_corpus_embed()
                if self.variant_type == 'fund':
                    self.x = self.corpus_embed
                    self.x_size = self.corpus_embed_size
                else:
                    self.x = tf.concat([self.corpus_embed, self.price], axis=2)
                    self.x_size = self.corpus_embed_size + self.price_size

    def _create_vmd_with_h_rec(self):
        with tf.name_scope('vmd'):
            with tf.variable_scope('vmd_h_rec'):
                x = tf.nn.dropout(self.x, keep_prob=1-self.dropout_vmd_in)
                x = tf.transpose(x, [1, 0, 2])  # max_n_days * batch_size * x_size
                y_ = tf.transpose(self.y_ph, [1, 0, 2])  # max_n_days * batch_size * y_size

                self.mask_aux_trading_days = tf.sequence_mask(self.T_ph - 1, self.max_n_days, dtype=tf.bool,
                                                              name='mask_aux_trading_days')

                def _loop_body(t, ta_h_s, ta_z_prior, ta_z_post, ta_kl):

                    with tf.variable_scope('iter_body', reuse=tf.AUTO_REUSE):

                        def _init():
                            h_s_init = tf.nn.tanh(tf.random_normal(shape=[self.batch_size, self.h_size]))
                            h_z_init = tf.nn.tanh(tf.random_normal(shape=[self.batch_size, self.z_size]))

                            z_init, _ = self._z(arg=h_z_init, is_prior=False)

                            return h_s_init, z_init

                        def _subsequent():
                            h_s_t_1 = tf.reshape(ta_h_s.read(t-1), [self.batch_size, self.h_size])
                            z_t_1 = tf.reshape(ta_z_post.read(t-1), [self.batch_size, self.z_size])

                            return h_s_t_1, z_t_1

                        h_s_t_1, z_t_1 = tf.cond(t >= 1, _subsequent, _init)

                        gate_args = [x[t], h_s_t_1, z_t_1]

                        with tf.variable_scope('gru_r'):
                            r = self._linear(gate_args, self.h_size, 'sigmoid')
                        with tf.variable_scope('gru_u'):
                            u = self._linear(gate_args, self.h_size, 'sigmoid')

                        h_args = [x[t], tf.multiply(r, h_s_t_1), z_t_1]

                        with tf.variable_scope('gru_h'):
                            h_tilde = self._linear(h_args, self.h_size, 'tanh')

                        h_s_t = tf.multiply(1 - u, h_s_t_1) + tf.multiply(u, h_tilde)

                        with tf.variable_scope('h_z_prior'):
                            h_z_prior_t = self._linear([x[t], h_s_t], self.z_size, 'tanh')
                        with tf.variable_scope('z_prior'):
                            z_prior_t, z_prior_t_pdf = self._z(h_z_prior_t, is_prior=True)

                        with tf.variable_scope('h_z_post'):
                            h_z_post_t = self._linear([x[t], h_s_t, y_[t]], self.z_size, 'tanh')
                        with tf.variable_scope('z_post'):
                            z_post_t, z_post_t_pdf = self._z(h_z_post_t, is_prior=False)

                    kl_t = ds.kl_divergence(z_post_t_pdf, z_prior_t_pdf)

                    # write
                    ta_h_s = ta_h_s.write(t, h_s_t)
                    ta_z_prior = ta_z_prior.write(t, z_prior_t)  # write: batch_size * z_size
                    ta_z_post = ta_z_post.write(t, z_post_t)  # write: batch_size * z_size
                    ta_kl = ta_kl.write(t, kl_t)  # write: batch_size * 1

                    return t + 1, ta_h_s, ta_z_prior, ta_z_post, ta_kl

                ta_h_s_init = tf.TensorArray(tf.float32, size=self.max_n_days, clear_after_read=False)
                ta_z_prior_init = tf.TensorArray(tf.float32, size=self.max_n_days)
                ta_z_post_init = tf.TensorArray(tf.float32, size=self.max_n_days, clear_after_read=False)
                ta_kl_init = tf.TensorArray(tf.float32, size=self.max_n_days)

                loop_init = (0, ta_h_s_init, ta_z_prior_init, ta_z_post_init, ta_kl_init)
                loop_cond = lambda t, *args: t < self.max_n_days
                _, ta_h_s, ta_z_prior, ta_z_post, ta_kl = tf.while_loop(loop_cond, _loop_body, loop_init)

                h_s = tf.reshape(ta_h_s.stack(), shape=(self.max_n_days, self.batch_size, self.h_size))
                z_shape = (self.max_n_days, self.batch_size, self.z_size)
                z_prior = tf.reshape(ta_z_prior.stack(), shape=z_shape)
                z_post = tf.reshape(ta_z_post.stack(), shape=z_shape)
                kl = tf.reshape(ta_kl.stack(), shape=z_shape)

                x = tf.transpose(x, [1, 0, 2])  # batch_size * max_n_days * x_size
                h_s = tf.transpose(h_s, [1, 0, 2])  # batch_size * max_n_days * vmd_h_size
                z_prior = tf.transpose(z_prior, [1, 0, 2])  # batch_size * max_n_days * z_size
                z_post = tf.transpose(z_post, [1, 0, 2])  # batch_size * max_n_days * z_size
                self.kl = tf.reduce_sum(tf.transpose(kl, [1, 0, 2]), axis=2)  # batch_size * max_n_days

                with tf.variable_scope('g'):
                    self.g = self._linear([x, h_s, z_post], self.g_size, 'tanh', use_bn=False)

                with tf.variable_scope('y'):
                    self.y = self._linear(self.g, self.y_size, 'softmax')

                sample_index = tf.reshape(tf.range(self.batch_size), (self.batch_size, 1), name='sample_index')
                self.indexed_T = tf.concat([sample_index, tf.reshape(self.T_ph-1, (self.batch_size, 1))], axis=1)

                def _infer_func():
                    g_T = tf.gather_nd(params=self.g, indices=self.indexed_T)  # batch_size * g_size

                    if not self.daily_att:
                        y_T = tf.gather_nd(params=self.y, indices=self.indexed_T)  # batch_size * y_size
                        return g_T, y_T

                    return g_T

                def _gen_func():
                    # use prior for g
                    z_prior_T = tf.gather_nd(params=z_prior, indices=self.indexed_T)  # batch_size * z_size
                    h_s_T = tf.gather_nd(params=h_s, indices=self.indexed_T)
                    x_T = tf.gather_nd(params=x, indices=self.indexed_T)

                    with tf.variable_scope('g', reuse=tf.AUTO_REUSE):
                        g_T = self._linear([x_T, h_s_T, z_prior_T], self.g_size, 'tanh', use_bn=False)

                    if not self.daily_att:
                        with tf.variable_scope('y', reuse=tf.AUTO_REUSE):
                            y_T = self._linear(g_T, self.y_size, 'softmax')
                        return g_T, y_T

                    return g_T

                if not self.daily_att:
                    self.g_T, self.y_T = tf.cond(tf.equal(self.is_training_phase, True), _infer_func, _gen_func)
                else:
                    self.g_T = tf.cond(tf.equal(self.is_training_phase, True), _infer_func, _gen_func)

    def _create_vmd_with_zh_rec(self):
        """
            Create a variational movement decoder.

            x: batch_size * max_n_days * vmd_in_size
            => vmd_h: batch_size * max_n_days * vmd_h_size
            => z: batch_size * max_n_days * vmd_z_size
            => y: batch_size * max_n_days * 2
        """
        with tf.name_scope('vmd'):
            with tf.variable_scope('vmd_zh_rec', reuse=tf.AUTO_REUSE):
                x = tf.nn.dropout(self.x, keep_prob=1-self.dropout_vmd_in)

                self.mask_aux_trading_days = tf.sequence_mask(self.T_ph - 1, self.max_n_days, dtype=tf.bool,
                                                              name='mask_aux_trading_days')

                if self.vmd_cell_type == 'ln-lstm':
                    cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.h_size)
                else:
                    cell = tf.contrib.rnn.GRUCell(self.h_size)
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0-self.dropout_vmd)

                init_state = None
                # calculate vmd_h, batch_size * max_n_days * vmd_h_size
                h_s, _ = tf.nn.dynamic_rnn(cell, x, sequence_length=self.T_ph, initial_state=init_state, dtype=tf.float32)

                # forward max_n_days
                x = tf.transpose(x, [1, 0, 2])  # max_n_days * batch_size * x_size
                h_s = tf.transpose(h_s, [1, 0, 2])  # max_n_days * batch_size * vmd_h_size
                y_ = tf.transpose(self.y_ph, [1, 0, 2])  # max_n_days * batch_size * y_size

                def _loop_body(t, ta_z_prior, ta_z_post, ta_kl):
                    """
                        iter body. iter over trading days.
                    """
                    with tf.variable_scope('iter_body', reuse=tf.AUTO_REUSE):

                        init = lambda: tf.random_normal(shape=[self.batch_size, self.z_size], name='z_post_t_1')
                        subsequent = lambda: tf.reshape(ta_z_post.read(t-1), [self.batch_size, self.z_size])

                        z_post_t_1 = tf.cond(t >= 1, subsequent, init)

                        with tf.variable_scope('h_z_prior'):
                            h_z_prior_t = self._linear([x[t], h_s[t], z_post_t_1], self.z_size, 'tanh')
                        with tf.variable_scope('z_prior'):
                            z_prior_t, z_prior_t_pdf = self._z(h_z_prior_t, is_prior=True)

                        with tf.variable_scope('h_z_post'):
                            h_z_post_t = self._linear([x[t], h_s[t], y_[t], z_post_t_1], self.z_size, 'tanh')
                        with tf.variable_scope('z_post'):
                            z_post_t, z_post_t_pdf = self._z(h_z_post_t, is_prior=False)

                    kl_t = ds.kl_divergence(z_post_t_pdf, z_prior_t_pdf)  # batch_size * z_size

                    ta_z_prior = ta_z_prior.write(t, z_prior_t)  # write: batch_size * z_size
                    ta_z_post = ta_z_post.write(t, z_post_t)  # write: batch_size * z_size
                    ta_kl = ta_kl.write(t, kl_t)  # write: batch_size * 1

                    return t + 1, ta_z_prior, ta_z_post, ta_kl

                # loop_init
                ta_z_prior_init = tf.TensorArray(tf.float32, size=self.max_n_days)
                ta_z_post_init = tf.TensorArray(tf.float32, size=self.max_n_days, clear_after_read=False)
                ta_kl_init = tf.TensorArray(tf.float32, size=self.max_n_days)

                loop_init = (0, ta_z_prior_init, ta_z_post_init, ta_kl_init)
                cond = lambda t, *args: t < self.max_n_days

                _, ta_z_prior, ta_z_post, ta_kl = tf.while_loop(cond, _loop_body, loop_init)

                z_shape = (self.max_n_days, self.batch_size, self.z_size)
                z_prior = tf.reshape(ta_z_prior.stack(), shape=z_shape)
                z_post = tf.reshape(ta_z_post.stack(), shape=z_shape)
                kl = tf.reshape(ta_kl.stack(), shape=z_shape)

                h_s = tf.transpose(h_s, [1, 0, 2])  # batch_size * max_n_days * vmd_h_size
                z_prior = tf.transpose(z_prior, [1, 0, 2])  # batch_size * max_n_days * z_size
                z_post = tf.transpose(z_post, [1, 0, 2])  # batch_size * max_n_days * z_size
                self.kl = tf.reduce_sum(tf.transpose(kl, [1, 0, 2]), axis=2)  # batch_size * max_n_days

                with tf.variable_scope('g'):
                    self.g = self._linear([h_s, z_post], self.g_size, 'tanh')  # batch_size * max_n_days * g_size

                with tf.variable_scope('y'):
                    self.y = self._linear(self.g, self.y_size, 'softmax')  # batch_size * max_n_days * y_size

                sample_index = tf.reshape(tf.range(self.batch_size), (self.batch_size, 1), name='sample_index')

                self.indexed_T = tf.concat([sample_index, tf.reshape(self.T_ph-1, (self.batch_size, 1))], axis=1)

                def _infer_func():
                    g_T = tf.gather_nd(params=self.g, indices=self.indexed_T)  # batch_size * g_size

                    if not self.daily_att:
                        y_T = tf.gather_nd(params=self.y, indices=self.indexed_T)  # batch_size * y_size
                        return g_T, y_T

                    return g_T

                def _gen_func():
                    # use prior for g & y
                    z_prior_T = tf.gather_nd(params=z_prior, indices=self.indexed_T)  # batch_size * z_size
                    h_s_T = tf.gather_nd(params=h_s, indices=self.indexed_T)

                    with tf.variable_scope('g', reuse=tf.AUTO_REUSE):
                        g_T = self._linear([h_s_T, z_prior_T], self.g_size, 'tanh', use_bn=False)

                    if not self.daily_att:
                        with tf.variable_scope('y', reuse=tf.AUTO_REUSE):
                            y_T = self._linear(g_T, self.y_size, 'softmax')
                        return g_T, y_T

                    return g_T

                if not self.daily_att:
                    self.g_T, self.y_T = tf.cond(tf.equal(self.is_training_phase, True), _infer_func, _gen_func)
                else:
                    self.g_T = tf.cond(tf.equal(self.is_training_phase, True), _infer_func, _gen_func)

    def _create_discriminative_vmd(self):
        """
            Create a discriminative movement decoder.

            x: batch_size * max_n_days * vmd_in_size
            => vmd_h: batch_size * max_n_days * vmd_h_size
            => z: batch_size * max_n_days * vmd_z_size
            => y: batch_size * max_n_days * 2
        """
        with tf.name_scope('vmd'):
            with tf.variable_scope('vmd_zh_rec', reuse=tf.AUTO_REUSE):
                x = tf.nn.dropout(self.x, keep_prob=1-self.dropout_vmd_in)

                self.mask_aux_trading_days = tf.sequence_mask(self.T_ph - 1, self.max_n_days, dtype=tf.bool,
                                                              name='mask_aux_trading_days')

                if self.vmd_cell_type == 'ln-lstm':
                    cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.h_size)
                else:
                    cell = tf.contrib.rnn.GRUCell(self.h_size)
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0-self.dropout_vmd)

                init_state = None
                h_s, _ = tf.nn.dynamic_rnn(cell, x, sequence_length=self.T_ph, initial_state=init_state, dtype=tf.float32)

                # forward max_n_days
                x = tf.transpose(x, [1, 0, 2])  # max_n_days * batch_size * x_size
                h_s = tf.transpose(h_s, [1, 0, 2])  # max_n_days * batch_size * vmd_h_size

                def _loop_body(t, ta_z):
                    """
                        iter body. iter over trading days.
                    """
                    with tf.variable_scope('iter_body', reuse=tf.AUTO_REUSE):

                        init = lambda: tf.random_normal(shape=[self.batch_size, self.z_size], name='z_post_t_1')
                        subsequent = lambda: tf.reshape(ta_z.read(t-1), [self.batch_size, self.z_size])

                        z_t_1 = tf.cond(t >= 1, subsequent, init)

                        with tf.variable_scope('h_z'):
                            h_z_t = self._linear([x[t], h_s[t], z_t_1], self.z_size, 'tanh')
                        with tf.variable_scope('z'):
                            z_t = self._linear(h_z_t, self.z_size, 'tanh')

                    ta_z = ta_z.write(t, z_t)  # write: batch_size * z_size
                    return t + 1, ta_z

                # loop_init
                ta_z_init = tf.TensorArray(tf.float32, size=self.max_n_days, clear_after_read=False)

                loop_init = (0, ta_z_init)
                cond = lambda t, *args: t < self.max_n_days

                _, ta_z_init = tf.while_loop(cond, _loop_body, loop_init)

                z_shape = (self.max_n_days, self.batch_size, self.z_size)
                z = tf.reshape(ta_z_init.stack(), shape=z_shape)

                h_s = tf.transpose(h_s, [1, 0, 2])  # batch_size * max_n_days * vmd_h_size
                z = tf.transpose(z, [1, 0, 2])  # batch_size * max_n_days * z_size

                with tf.variable_scope('g'):
                    self.g = self._linear([h_s, z], self.g_size, 'tanh')  # batch_size * max_n_days * g_size

                with tf.variable_scope('y'):
                    self.y = self._linear(self.g, self.y_size, 'softmax')  # batch_size * max_n_days * y_size

                # get g_T
                sample_index = tf.reshape(tf.range(self.batch_size), (self.batch_size, 1), name='sample_index')
                self.indexed_T = tf.concat([sample_index, tf.reshape(self.T_ph-1, (self.batch_size, 1))], axis=1)
                self.g_T = tf.gather_nd(params=self.g, indices=self.indexed_T)

    def _build_vmd(self):
        if self.variant_type == 'discriminative':
            self._create_discriminative_vmd()
        else:
            if self.vmd_rec == 'h':
                self._create_vmd_with_h_rec()
            else:
                self._create_vmd_with_zh_rec()

    def _build_temporal_att(self):
        """
            g: batch_size * max_n_days * g_size
            g_T: batch_size * g_size
        """
        with tf.name_scope('tda'):
            with tf.variable_scope('tda'):
                with tf.variable_scope('v_i'):
                    proj_i = self._linear([self.g], self.g_size, 'tanh', use_bias=False)
                    w_i = tf.get_variable('w_i', shape=(self.g_size, 1), initializer=self.initializer)
                v_i = tf.reduce_sum(tf.tensordot(proj_i, w_i, axes=1), axis=-1)  # batch_size * max_n_days

                with tf.variable_scope('v_d'):
                    proj_d = self._linear([self.g], self.g_size, 'tanh', use_bias=False)
                g_T = tf.expand_dims(self.g_T, axis=-1)  # batch_size * g_size * 1
                v_d = tf.reduce_sum(tf.matmul(proj_d, g_T), axis=-1)  # batch_size * max_n_days

                aux_score = tf.multiply(v_i, v_d, name='v_stared')
                ninf = tf.fill(tf.shape(aux_score), np.NINF)
                masked_aux_score = tf.where(self.mask_aux_trading_days, aux_score, ninf)
                v_stared = tf.nn.softmax(masked_aux_score)

                # v_stared: batch_size * max_n_days
                self.v_stared = tf.where(tf.is_nan(v_stared), tf.zeros_like(v_stared), v_stared)

                if self.daily_att == 'y':
                    context = tf.transpose(self.y, [0, 2, 1])  # batch_size * y_size * max_n_days
                else:
                    context = tf.transpose(self.g, [0, 2, 1])  # batch_size * g_size * max_n_days

                v_stared = tf.expand_dims(self.v_stared, -1)  # batch_size * max_n_days * 1
                att_c = tf.reduce_sum(tf.matmul(context, v_stared), axis=-1)  # batch_size * g_size / y_size
                with tf.variable_scope('y_T'):
                    self.y_T = self._linear([att_c, self.g_T], self.y_size, 'softmax')

    def _create_generative_ata(self):
        """
             calculate loss.

             g: batch_size * max_n_days * g_size
             y: batch_size * max_n_days * y_size
             kl_loss: batch_size * max_n_days
             => loss: batch_size
        """
        with tf.name_scope('ata'):
            with tf.variable_scope('ata'):
                v_aux = self.alpha * self.v_stared  # batch_size * max_n_days

                minor = 0.0  # 0.0, 1e-7
                likelihood_aux = tf.reduce_sum(tf.multiply(self.y_ph, tf.log(self.y + minor)), axis=2)  # batch_size * max_n_days

                kl_lambda = self._kl_lambda()
                obj_aux = likelihood_aux - kl_lambda * self.kl  # batch_size * max_n_days

                # deal with T specially, likelihood_T: batch_size, 1
                self.y_T_ = tf.gather_nd(params=self.y_ph, indices=self.indexed_T)  # batch_size * y_size
                likelihood_T = tf.reduce_sum(tf.multiply(self.y_T_, tf.log(self.y_T + minor)), axis=1, keep_dims=True)

                kl_T = tf.reshape(tf.gather_nd(params=self.kl, indices=self.indexed_T), shape=[self.batch_size, 1])
                obj_T = likelihood_T - kl_lambda * kl_T

                obj = obj_T + tf.reduce_sum(tf.multiply(obj_aux, v_aux), axis=1, keep_dims=True)  # batch_size * 1
                self.loss = tf.reduce_mean(-obj, axis=[0, 1])

    def _create_discriminative_ata(self):
        """
             calculate discriminative loss.

             g: batch_size * max_n_days * g_size
             y: batch_size * max_n_days * y_size
             => loss: batch_size
        """
        with tf.name_scope('ata'):
            with tf.variable_scope('ata'):
                v_aux = self.alpha * self.v_stared  # batch_size * max_n_days

                minor = 0.0  # 0.0, 1e-7
                likelihood_aux = tf.reduce_sum(tf.multiply(self.y_ph, tf.log(self.y + minor)), axis=2)  # batch_size * max_n_days

                # deal with T specially, likelihood_T: batch_size, 1
                self.y_T_ = tf.gather_nd(params=self.y_ph, indices=self.indexed_T)  # batch_size * y_size
                likelihood_T = tf.reduce_sum(tf.multiply(self.y_T_, tf.log(self.y_T + minor)), axis=1, keep_dims=True)

                obj = likelihood_T + tf.reduce_sum(tf.multiply(likelihood_aux, v_aux), axis=1, keep_dims=True)  # batch_size * 1
                self.loss = tf.reduce_mean(-obj, axis=[0, 1])

    def _build_ata(self):
        if self.variant_type == 'discriminative':
            self._create_discriminative_ata()
        else:
            self._create_generative_ata()

    def _create_optimizer(self):
        with tf.name_scope('optimizer'):
            if self.opt == 'sgd':
                decayed_lr = tf.train.exponential_decay(learning_rate=self.lr, global_step=self.global_step,
                                                        decay_steps=self.decay_step, decay_rate=self.decay_rate)
                optimizer = tf.train.MomentumOptimizer(learning_rate=decayed_lr, momentum=self.momentum)
            else:
                optimizer = tf.train.AdamOptimizer(self.lr)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip)
            self.optimize = optimizer.apply_gradients(zip(gradients, variables))
            self.global_step = tf.assign_add(self.global_step, 1)

    def assemble_graph(self):
        logger.info('Start graph assembling...')
        with tf.device('/device:GPU:0'):
            self._build_placeholders()
            self._build_embeds()
            self._build_mie()
            self._build_vmd()
            self._build_temporal_att()
            self._build_ata()
            self._create_optimizer()

    def _kl_lambda(self):
        def _nonzero_kl_lambda():
            if self.use_constant_kl_lambda:
                return self.constant_kl_lambda
            else:
                return tf.minimum(self.kl_lambda_anneal_rate * global_step, 1.0)

        global_step = tf.cast(self.global_step, tf.float32)

        return tf.cond(global_step < self.kl_lambda_start_step, lambda: 0.0, _nonzero_kl_lambda)

    def _linear(self, args, output_size, activation=None, use_bias=True, use_bn=False):
        if type(args) not in (list, tuple):
            args = [args]

        shape = [a if a else -1 for a in args[0].get_shape().as_list()[:-1]]
        shape.append(output_size)

        sizes = [a.get_shape()[-1].value for a in args]
        total_arg_size = sum(sizes)
        scope = tf.get_variable_scope()
        x = args[0] if len(args) == 1 else tf.concat(args, -1)

        with tf.variable_scope(scope):
            weight = tf.get_variable('weight', [total_arg_size, output_size], dtype=tf.float32, initializer=self.initializer)
            res = tf.tensordot(x, weight, axes=1)
            if use_bias:
                bias = tf.get_variable('bias', [output_size], dtype=tf.float32, initializer=self.bias_initializer)
                res = tf.nn.bias_add(res, bias)

        res = tf.reshape(res, shape)

        if use_bn:
            res = batch_norm(res, center=True, scale=True, decay=0.99, updates_collections=None,
                             is_training=self.is_training_phase, scope=scope)

        if activation == 'tanh':
            res = tf.nn.tanh(res)
        elif activation == 'sigmoid':
            res = tf.nn.sigmoid(res)
        elif activation == 'relu':
            res = tf.nn.relu(res)
        elif activation == 'softmax':
            res = tf.nn.softmax(res)

        return res

    def _z(self, arg, is_prior):
        mean = self._linear(arg, self.z_size)
        stddev = self._linear(arg, self.z_size)
        stddev = tf.sqrt(tf.exp(stddev))
        epsilon = tf.random_normal(shape=[self.batch_size, self.z_size])

        z = mean if is_prior else mean + tf.multiply(stddev, epsilon)
        pdf_z = ds.Normal(loc=mean, scale=stddev)

        return z, pdf_z
