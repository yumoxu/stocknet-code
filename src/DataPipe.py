#!/usr/local/bin/python
import os
import io
import json
import numpy as np
from datetime import datetime, timedelta
import random
from ConfigLoader import logger, path_parser, config_model, dates, stock_symbols, vocab, vocab_size


class DataPipe:

    def __init__(self):
        # load path
        self.movement_path = path_parser.movement
        self.tweet_path = path_parser.preprocessed
        self.vocab_path = path_parser.vocab
        self.glove_path = path_parser.glove

        # load dates
        self.train_start_date = dates['train_start_date']
        self.train_end_date = dates['train_end_date']
        self.dev_start_date = dates['dev_start_date']
        self.dev_end_date = dates['dev_end_date']
        self.test_start_date = dates['test_start_date']
        self.test_end_date = dates['test_end_date']

        # load model config
        self.batch_size = config_model['batch_size']
        self.shuffle = config_model['shuffle']

        self.max_n_days = config_model['max_n_days']
        self.max_n_words = config_model['max_n_words']
        self.max_n_msgs = config_model['max_n_msgs']

        self.word_embed_type = config_model['word_embed_type']
        self.word_embed_size = config_model['word_embed_size']
        self.stock_embed_size = config_model['stock_embed_size']
        self.init_stock_with_word= config_model['init_stock_with_word']
        self.price_embed_size = config_model['word_embed_size']
        self.y_size = config_model['y_size']

        assert self.word_embed_type in ('rand', 'glove')

    @staticmethod
    def _convert_token_to_id(token, token_id_dict):
        if token not in token_id_dict:
            token = 'UNK'
        return token_id_dict[token]

    def _get_start_end_date(self, phase):
        """
            phase: train, dev, test, unit_test
            => start_date & end_date
        """
        assert phase in {'train', 'dev', 'test', 'whole', 'unit_test'}
        if phase == 'train':
            return self.train_start_date, self.train_end_date
        elif phase == 'dev':
            return self.dev_start_date, self.dev_end_date
        elif phase == 'test':
            return self.test_start_date, self.test_end_date
        elif phase == 'whole':
            return self.train_start_date, self.test_end_date
        else:
            return '2012-07-23', '2012-08-05'  # '2014-07-23', '2014-08-05'

    def _get_batch_size(self, phase):
        """
            phase: train, dev, test, unit_test
        """
        if phase == 'train':
            return self.batch_size
        elif phase == 'unit_test':
            return 5
        else:
            return 1

    def index_token(self, token_list, key='id', type='word'):
        assert key in ('id', 'token')
        assert type in ('word', 'stock')
        indexed_token_dict = dict()

        if type == 'word':
            token_list_cp = list(token_list)  # un-change the original input
            token_list_cp.insert(0, 'UNK')  # for unknown tokens
        else:
            token_list_cp = token_list

        if key == 'id':
            for id in range(len(token_list_cp)):
                indexed_token_dict[id] = token_list_cp[id]
        else:
            for id in range(len(token_list_cp)):
                indexed_token_dict[token_list_cp[id]] = id

        # id_token_dict = dict(zip(token_id_dict.values(), token_id_dict.keys()))
        return indexed_token_dict

    def build_stock_id_word_id_dict(self):
        # load vocab, user, stock list
        stock_id_word_id_dict = dict()

        vocab_id_dict = self.index_token(vocab, key='token')
        id_stock_dict = self.index_token(stock_symbols, type='stock')

        for (stock_id, stock_symbol) in id_stock_dict.items():
            stock_symbol = stock_symbol.lower()
            if stock_symbol in vocab_id_dict:
                stock_id_word_id_dict[stock_id] = vocab_id_dict[stock_symbol]
            else:
                stock_id_word_id_dict[stock_id] = None
        return stock_id_word_id_dict

    def _convert_words_to_ids(self, words, vocab_id_dict):
        """
            Replace each word in the data set with its index in the dictionary

        :param words: words in tweet
        :param vocab_id_dict: dict, vocab-id
        :return:
        """
        return [self._convert_token_to_id(w, vocab_id_dict) for w in words]

    def _get_prices_and_ts(self, ss, main_target_date):

        def _get_mv_class(data, use_one_hot=False):
            mv = float(data[1])
            if self.y_size == 2:
                if mv <= 1e-7:
                    return [1.0, 0.0] if use_one_hot else 0
                else:
                    return [0.0, 1.0] if use_one_hot else 1

            if self.y_size == 3:
                threshold_1, threshold_2 = -0.004, 0.005
                if mv < threshold_1:
                    return [1.0, 0.0, 0.0] if use_one_hot else 0
                elif mv < threshold_2:
                    return [0.0, 1.0, 0.0] if use_one_hot else 1
                else:
                    return [0.0, 0.0, 1.0] if use_one_hot else 2

        def _get_y(data):
            return _get_mv_class(data, use_one_hot=True)

        def _get_prices(data):
            return [float(p) for p in data[3:6]]

        def _get_mv_percents(data):
            return _get_mv_class(data)

        ts, ys, prices, mv_percents, main_mv_percent = list(), list(), list(), list(), 0.0
        d_t_min = main_target_date - timedelta(days=self.max_n_days-1)

        stock_movement_path = os.path.join(str(self.movement_path), '{}.txt'.format(ss))
        with io.open(stock_movement_path, 'r', encoding='utf8') as movement_f:
            for line in movement_f:  # descend
                data = line.split('\t')
                t = datetime.strptime(data[0], '%Y-%m-%d').date()
                # logger.info(t)
                if t == main_target_date:
                    # logger.info(t)
                    ts.append(t)
                    ys.append(_get_y(data))
                    main_mv_percent = data[1]
                    if -0.005 <= float(main_mv_percent) < 0.0055:  # discard sample with low movement percent
                        return None
                if d_t_min <= t < main_target_date:
                    ts.append(t)
                    ys.append(_get_y(data))
                    prices.append(_get_prices(data))  # high, low, close
                    mv_percents.append(_get_mv_percents(data))
                if t < d_t_min:  # one additional line for x_1_prices. not a referred trading day
                    prices.append(_get_prices(data))
                    mv_percents.append(_get_mv_percents(data))
                    break

        T = len(ts)
        if len(ys) != T or len(prices) != T or len(mv_percents) != T:  # ensure data legibility
            return None

        # ascend
        for item in (ts, ys, mv_percents, prices):
            item.reverse()

        prices_and_ts = {
            'T': T,
            'ts': ts,
            'ys': ys,
            'main_mv_percent': main_mv_percent,
            'mv_percents': mv_percents,
            'prices': prices,
        }

        return prices_and_ts

    def _get_unaligned_corpora(self, ss, main_target_date, vocab_id_dict):
        def get_ss_index(word_seq, ss):
            ss = ss.lower()
            ss_index = len(word_seq) - 1  # init
            if ss in word_seq:
                ss_index = word_seq.index(ss)
            else:
                if '$' in word_seq:
                    dollar_index = word_seq.index('$')
                    if dollar_index is not len(word_seq) - 1 and ss in word_seq[dollar_index + 1]:
                        ss_index = dollar_index + 1
                    else:
                        for index in range(dollar_index + 1, len(word_seq)):
                            if ss in word_seq[index]:
                                ss_index = index
                                break
            return ss_index

        unaligned_corpora = list()  # list of sets: (d, msgs, ss_indices)
        stock_tweet_path = os.path.join(str(self.tweet_path), ss)

        d_d_max = main_target_date - timedelta(days=1)
        d_d_min = main_target_date - timedelta(days=self.max_n_days)

        d = d_d_max  # descend
        while d >= d_d_min:
            msg_fp = os.path.join(stock_tweet_path, d.isoformat())
            if os.path.exists(msg_fp):
                word_mat = np.zeros([self.max_n_msgs, self.max_n_words], dtype=np.int32)
                n_word_vec = np.zeros([self.max_n_msgs, ], dtype=np.int32)
                ss_index_vec = np.zeros([self.max_n_msgs, ], dtype=np.int32)
                msg_id = 0
                with open(msg_fp, 'r') as tweet_f:
                    for line in tweet_f:
                        msg_dict = json.loads(line)
                        text = msg_dict['text']
                        if not text:
                            continue

                        words = text[:self.max_n_words]
                        word_ids = self._convert_words_to_ids(words, vocab_id_dict)
                        n_words = len(word_ids)

                        n_word_vec[msg_id] = n_words
                        word_mat[msg_id, :n_words] = word_ids
                        ss_index_vec[msg_id] = get_ss_index(words, ss)

                        msg_id += 1
                        if msg_id == self.max_n_msgs:
                            break
                corpus = [d, word_mat[:msg_id], ss_index_vec[:msg_id], n_word_vec[:msg_id], msg_id]
                unaligned_corpora.append(corpus)
            d -= timedelta(days=1)

        unaligned_corpora.reverse()  # ascend
        return unaligned_corpora

    def _trading_day_alignment(self, ts, T, unaligned_corpora):
        aligned_word_tensor = np.zeros([T, self.max_n_msgs, self.max_n_words], dtype=np.int32)
        aligned_ss_index_mat = np.zeros([T, self.max_n_msgs], dtype=np.int32)
        aligned_n_words_mat = np.zeros([T, self.max_n_msgs], dtype=np.int32)
        aligned_n_msgs_vec = np.zeros([T, ], dtype=np.int32)

        # list for gathering
        aligned_msgs = [[] for _ in range(T)]
        aligned_ss_indices = [[] for _ in range(T)]
        aligned_n_words = [[] for _ in range(T)]
        aligned_n_msgs = [[] for _ in range(T)]

        corpus_t_indices = []
        max_threshold = 0

        for corpus in unaligned_corpora:
            d = corpus[0]
            for t in range(T):
                if d < ts[t]:
                    corpus_t_indices.append(t)
                    break

        assert len(corpus_t_indices) == len(unaligned_corpora)

        for i in range(len(unaligned_corpora)):
            corpus, t = unaligned_corpora[i], corpus_t_indices[i]
            word_mat, ss_index_vec, n_word_vec, n_msgs = corpus[1:]
            aligned_msgs[t].extend(word_mat)
            aligned_ss_indices[t].extend(ss_index_vec)
            aligned_n_words[t].append(n_word_vec)
            aligned_n_msgs[t].append(n_msgs)

        def is_eligible():
            n_fails = len([0 for n_msgs in aligned_n_msgs if sum(n_msgs) == 0])
            return n_fails <= max_threshold

        if not is_eligible():
            return None

        # gather into nd_array and clip exceeded part
        for t in range(T):
            n_msgs = sum(aligned_n_msgs[t])

            if aligned_msgs[t] and aligned_ss_indices[t] and aligned_n_words[t]:
                msgs, ss_indices, n_word = np.vstack(aligned_msgs[t]), np.hstack(aligned_ss_indices[t]), np.hstack(aligned_n_words[t])
                assert len(msgs) == len(ss_indices) == len(n_word)
                n_msgs = min(n_msgs, self.max_n_msgs)  # clip length
                aligned_n_msgs_vec[t] = n_msgs
                aligned_word_tensor[t, :n_msgs] = msgs[:n_msgs]
                aligned_ss_index_mat[t, :n_msgs] = ss_indices[:n_msgs]
                aligned_n_words_mat[t, :n_msgs] = n_word[:n_msgs]

        aligned_info_dict = {
            'msgs': aligned_word_tensor,
            'ss_indices': aligned_ss_index_mat,
            'n_words': aligned_n_words_mat,
            'n_msgs': aligned_n_msgs_vec,
        }

        return aligned_info_dict

    def sample_gen_from_one_stock(self, vocab_id_dict, stock_id_dict, s, phase):
        """
            generate samples for the given stock.

            => tuple, (x:dict, y_:int, price_seq: list of floats, prediction_date_str:str)
        """
        start_date, end_date = self._get_start_end_date(phase)
        stock_movement_path = os.path.join(str(self.movement_path), '{}.txt'.format(s))
        main_target_dates = []

        with open(stock_movement_path, 'r') as movement_f:
            for line in movement_f:
                data = line.split('\t')
                main_target_date = datetime.strptime(data[0], '%Y-%m-%d').date()
                main_target_date_str = main_target_date.isoformat()

                if start_date <= main_target_date_str < end_date:
                    main_target_dates.append(main_target_date)

        if self.shuffle:  # shuffle data
            random.shuffle(main_target_dates)

        for main_target_date in main_target_dates:
            # logger.info('start _get_unaligned_corpora')
            unaligned_corpora = self._get_unaligned_corpora(s, main_target_date, vocab_id_dict)
            # logger.info('start _get_prices_and_ts')
            prices_and_ts = self._get_prices_and_ts(s, main_target_date)
            if not prices_and_ts:
                continue

            # logger.info('start _trading_day_alignment')
            aligned_info_dict = self._trading_day_alignment(prices_and_ts['ts'], prices_and_ts['T'], unaligned_corpora)
            if not aligned_info_dict:
                continue

            sample_dict = {
                # meta info
                'stock': self._convert_token_to_id(s, stock_id_dict),
                'main_target_date': main_target_date.isoformat(),
                'T': prices_and_ts['T'],
                # target
                'ys': prices_and_ts['ys'],
                'main_mv_percent': prices_and_ts['main_mv_percent'],
                'mv_percents': prices_and_ts['mv_percents'],
                # source
                'prices': prices_and_ts['prices'],
                'msgs': aligned_info_dict['msgs'],
                'ss_indices': aligned_info_dict['ss_indices'],
                'n_words': aligned_info_dict['n_words'],
                'n_msgs': aligned_info_dict['n_msgs'],
            }

            yield sample_dict

    def batch_gen(self, phase):
        batch_size = self._get_batch_size(phase)
        # prepare vocab, user, stock dict
        vocab_id_dict = self.index_token(vocab, key='token')
        stock_id_dict = self.index_token(stock_symbols, key='token', type='stock')
        generators = [self.sample_gen_from_one_stock(vocab_id_dict, stock_id_dict, s, phase) for s in stock_symbols]
        # logger.info('{0} Generators prepared...'.format(len(generators)))

        while True:
            # start_time = time.time()
            # logger.info('start to collect a batch...')
            stock_batch = np.zeros([batch_size, ], dtype=np.int32)
            T_batch = np.zeros([batch_size, ], dtype=np.int32)
            y_batch = np.zeros([batch_size, self.max_n_days, self.y_size], dtype=np.float32)
            main_mv_percent_batch = np.zeros([batch_size, ], dtype=np.float32)
            mv_percent_batch = np.zeros([batch_size, self.max_n_days], dtype=np.float32)
            price_batch = np.zeros([batch_size, self.max_n_days, 3], dtype=np.float32)
            word_batch = np.zeros([batch_size, self.max_n_days, self.max_n_msgs, self.max_n_words], dtype=np.int32)
            ss_index_batch = np.zeros([batch_size, self.max_n_days, self.max_n_msgs], dtype=np.int32)
            n_msgs_batch = np.zeros([batch_size, self.max_n_days], dtype=np.int32)
            n_words_batch = np.zeros([batch_size, self.max_n_days, self.max_n_msgs], dtype=np.int32)

            sample_id = 0
            while sample_id < batch_size:
                gen_id = random.randint(0, len(generators)-1)
                try:
                    sample_dict = next(generators[gen_id])
                    T = sample_dict['T']
                    # meta
                    stock_batch[sample_id] = sample_dict['stock']
                    T_batch[sample_id] = T
                    # target
                    y_batch[sample_id, :T] = sample_dict['ys']
                    main_mv_percent_batch[sample_id] = sample_dict['main_mv_percent']
                    mv_percent_batch[sample_id, :T] = sample_dict['mv_percents']
                    # source
                    price_batch[sample_id, :T] = sample_dict['prices']
                    word_batch[sample_id, :T] = sample_dict['msgs']
                    ss_index_batch[sample_id, :T] = sample_dict['ss_indices']
                    n_msgs_batch[sample_id, :T] = sample_dict['n_msgs']
                    n_words_batch[sample_id, :T] = sample_dict['n_words']

                    sample_id += 1
                except StopIteration:
                    del generators[gen_id]
                    if generators:
                        continue
                    else:
                        raise StopIteration

            batch_dict = {
                # meta
                'batch_size': sample_id,
                'stock_batch': stock_batch,
                'T_batch': T_batch,
                # target
                'y_batch': y_batch,
                'main_mv_percent_batch': main_mv_percent_batch,
                'mv_percent_batch': mv_percent_batch,
                # source
                'price_batch': price_batch,
                'word_batch': word_batch,
                'ss_index_batch': ss_index_batch,
                'n_msgs_batch': n_msgs_batch,
                'n_words_batch': n_words_batch,
            }

            yield batch_dict

    def batch_gen_by_stocks(self, phase):
        batch_size = 2000
        vocab_id_dict = self.index_token(vocab, key='token')
        stock_id_dict = self.index_token(stock_symbols, key='token', type='stock')

        for s in stock_symbols:
            gen = self.sample_gen_from_one_stock(vocab_id_dict, stock_id_dict, s, phase)

            stock_batch = np.zeros([batch_size, ], dtype=np.int32)
            T_batch = np.zeros([batch_size, ], dtype=np.int32)
            n_msgs_batch = np.zeros([batch_size, self.max_n_days], dtype=np.int32)
            n_words_batch = np.zeros([batch_size, self.max_n_days, self.max_n_msgs], dtype=np.int32)
            y_batch = np.zeros([batch_size, self.max_n_days, self.y_size], dtype=np.float32)
            price_batch = np.zeros([batch_size, self.max_n_days, 3], dtype=np.float32)
            mv_percent_batch = np.zeros([batch_size, self.max_n_days], dtype=np.float32)
            word_batch = np.zeros([batch_size, self.max_n_days, self.max_n_msgs, self.max_n_words], dtype=np.int32)
            ss_index_batch = np.zeros([batch_size, self.max_n_days, self.max_n_msgs], dtype=np.int32)
            main_mv_percent_batch = np.zeros([batch_size, ], dtype=np.float32)

            sample_id = 0
            while True:
                try:
                    sample_info_dict = next(gen)
                    T = sample_info_dict['T']

                    # meta
                    stock_batch[sample_id] = sample_info_dict['stock']
                    T_batch[sample_id] = sample_info_dict['T']
                    # target
                    y_batch[sample_id, :T] = sample_info_dict['ys']
                    main_mv_percent_batch[sample_id] = sample_info_dict['main_mv_percent']
                    mv_percent_batch[sample_id, :T] = sample_info_dict['mv_percents']
                    # source
                    price_batch[sample_id, :T] = sample_info_dict['prices']
                    word_batch[sample_id, :T] = sample_info_dict['msgs']
                    ss_index_batch[sample_id, :T] = sample_info_dict['ss_indices']
                    n_msgs_batch[sample_id, :T] = sample_info_dict['n_msgs']
                    n_words_batch[sample_id, :T] = sample_info_dict['n_words']

                    sample_id += 1
                except StopIteration:
                    break

            n_sample_threshold = 1
            if sample_id < n_sample_threshold:
                continue

            batch_dict = {
                # meta
                's': s,
                'batch_size': sample_id,
                'stock_batch': stock_batch[:sample_id],
                'T_batch': T_batch[:sample_id],
                # target
                'y_batch': y_batch[:sample_id],
                'main_mv_percent_batch': main_mv_percent_batch[:sample_id],
                'mv_percent_batch': mv_percent_batch[:sample_id],
                # source
                'price_batch': price_batch[:sample_id],
                'word_batch': word_batch[:sample_id],
                'ss_index_batch': ss_index_batch[:sample_id],
                'n_msgs_batch': n_msgs_batch[:sample_id],
                'n_words_batch': n_words_batch[:sample_id],
            }

            yield batch_dict

    def sample_mv_percents(self, phase):
        main_mv_percents = []
        for s in stock_symbols:
            start_date, end_date = self._get_start_end_date(phase)
            stock_mv_path = os.path.join(str(self.movement_path), '{}.txt'.format(s))
            main_target_dates = []

            with open(stock_mv_path, 'r') as movement_f:
                for line in movement_f:
                    data = line.split('\t')
                    main_target_date = datetime.strptime(data[0], '%Y-%m-%d').date()
                    main_target_date_str = main_target_date.isoformat()

                    if start_date <= main_target_date_str < end_date:
                        main_target_dates.append(main_target_date)

            for main_target_date in main_target_dates:
                prices_and_ts = self._get_prices_and_ts(s, main_target_date)
                if not prices_and_ts:
                    continue
                main_mv_percents.append(prices_and_ts['main_mv_percent'])

            logger.info('finished: {}'.format(s))

        return main_mv_percents

    def init_word_table(self):
        word_table_init = np.random.random((vocab_size, self.word_embed_size)) * 2 - 1  # [-1.0, 1.0]

        if self.word_embed_type is not 'rand':
            n_replacement = 0
            vocab_id_dict = self.index_token(vocab, key='token')

            with io.open(self.glove_path, 'r', encoding='utf-8') as f:
                for line in f:
                    tuples = line.split()
                    word, embed = tuples[0], [float(embed_col) for embed_col in tuples[1:]]
                    if word in ['<unk>', 'unk']:  # unify UNK
                        word = 'UNK'
                    if word in vocab_id_dict:
                        n_replacement += 1
                        word_id = vocab_id_dict[word]
                        word_table_init[word_id] = embed

            logger.info('ASSEMBLE: word table #replacement: {}'.format(n_replacement))
        return word_table_init
