#!/usr/local/bin/python
import metrics as metrics
from ConfigLoader import logger


def print_batch_stat(n_iter, train_batch_loss, train_batch_n_acc, train_batch_size):
    iter_str = '\titer: {0}'.format(n_iter)
    loss_str = 'batch loss: {:.6f}'.format(train_batch_loss) if type(train_batch_loss) is float else 'batch loss: {}'.format(train_batch_loss)
    train_batch_acc = metrics.eval_acc(n_acc=train_batch_n_acc, total=train_batch_size)
    acc_str = 'batch acc: {:.6f}'.format(train_batch_acc)
    logger.info(', '.join((iter_str, loss_str, acc_str)))


def print_epoch_stat(epoch_loss, epoch_acc):
    epoch_stat_pattern = 'Epoch: loss: {0:.6f}, acc: {1:.6f}'
    logger.info(epoch_stat_pattern.format(epoch_loss, epoch_acc))


def print_eval_res(result_dict, use_mcc=None):
    eval_loss, eval_acc = result_dict['loss'], result_dict['acc']
    iter_str = '\tEval'
    loss_str = 'loss: {:.6f}'.format(eval_loss) if type(eval_loss) is float else 'eval loss: {}'.format(eval_loss)
    acc_str = 'acc: {:.6f}'.format(eval_acc)
    info_list = [iter_str, loss_str, acc_str]
    if use_mcc:
        mcc = result_dict['mcc']
        mcc_str = 'mcc: {:.6f}'.format(mcc) if mcc else 'mcc: {}'.format(mcc)
        info_list.append(mcc_str)
    logger.info(', '.join(info_list))
