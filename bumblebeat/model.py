from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import time

import tensorflow as tf

import numpy as np

import bumblebeat.data
from bumblebeat.gpu_utils import assign_to_gpu, average_grads_and_vars
import bumblebeat.transformer as transformer


def get_model_fn(n_token, cutoffs, conf):
    def model_fn(inp, tgt, mems, is_training, conf):
        inp = tf.transpose(inp, [1, 0])
        tgt = tf.transpose(tgt, [1, 0])

        if conf['init'] == "uniform":
          initializer = tf.initializers.random_uniform(
              minval=-conf['init_range'],
              maxval=conf['init_range'],
              seed=None)
        elif conf['init'] == "normal":
          initializer = tf.initializers.random_normal(
              stddev=conf['init_std'],
              seed=None)
          proj_initializer = tf.initializers.random_normal(
              stddev=conf['proj_init_std'],
              seed=None)

        tie_projs = [False for _ in range(len(cutoffs) + 1)]
        if conf['proj_share_all_but_first']:
          for i in range(1, len(tie_projs)):
            tie_projs[i] = True

        loss, new_mems = transformer.transformer(
            dec_inp=inp,
            target=tgt,
            mems=mems,
            n_token=n_token,
            n_layer=conf['n_layer'],
            d_model=conf['d_model'],
            d_embed=conf['d_embed'],
            n_head=conf['n_head'],
            d_head=conf['d_head'],
            d_inner=conf['d_inner'],
            dropout=conf['dropout'],
            dropatt=conf['dropatt'],
            initializer=initializer,
            proj_initializer=proj_initializer,
            is_training=is_training,
            mem_len=conf['mem_len'],
            cutoffs=cutoffs,
            div_val=conf['div_val'],
            tie_projs=tie_projs,
            input_perms=None,
            target_perms=None,
            head_target=None,
            same_length=conf['same_length'],
            clamp_len=conf['clamp_len'],
            use_tpu=False,
            untie_r=conf['untie_r'],
            proj_same_dim=conf['proj_same_dim'])

        # number of parameters
        num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
        tf.logging.info('#params: {}'.format(num_params))

        # format_str = '{{:<{0}s}}\t{{}}'.format(
        #     max([len(v.name) for v in tf.trainable_variables()]))
        # for v in tf.trainable_variables():
        #   tf.logging.info(format_str.format(v.name, v.get_shape()))

        if is_training:
          all_vars = tf.trainable_variables()
          grads = tf.gradients(loss, all_vars)
          grads_and_vars = list(zip(grads, all_vars))

          return loss, new_mems, grads_and_vars
        else:
          return loss, new_mems

    return model_fn


def single_core_graph(n_token, cutoffs, is_training, inp, tgt, mems, conf):
    model_fn = get_model_fn(
        n_token=n_token,
        cutoffs=cutoffs,
        conf=conf)

    model_ret = model_fn(
        inp=inp,
        tgt=tgt,
        mems=mems,
        is_training=is_training,
        conf=conf)

    return model_ret


def train(n_token, cutoffs, ps_device, conf):
    ##### Get input function and model function
    train_input_fn, train_record_info = bumblebeat.data.get_input_fn(
        record_info_dir=conf['record_info_dir'],
        split="train",
        per_host_bsz=conf['train_batch_size'],
        tgt_len=conf['tgt_len'],
        num_core_per_host=conf['num_core_per_host'],
        num_hosts=1,
        use_tpu=False)

    tf.logging.info("num of batches {}".format(train_record_info["num_batch"]))

    ##### Create computational graph
    train_set = train_input_fn({
        "batch_size": conf['train_batch_size'],
        "data_dir": conf['data_dir']})

    input_feed, label_feed = train_set.make_one_shot_iterator().get_next()

    inputs = tf.split(input_feed, conf['num_core_per_host'], 0)
    labels = tf.split(label_feed, conf['num_core_per_host'], 0)

    per_core_bsz = conf['train_batch_size'] // conf['num_core_per_host']

    tower_mems, tower_losses, tower_new_mems, tower_grads_and_vars = [], [], [], []

    for i in range(conf['num_core_per_host']):
      reuse = True if i > 0 else None
      with tf.device(assign_to_gpu(i, ps_device)), \
          tf.variable_scope(tf.get_variable_scope(), reuse=reuse):

        mems_i = [tf.placeholder(tf.float32,
                                 [conf['mem_len'], per_core_bsz, conf['d_model']])
                  for _ in range(conf['n_layer'])]

        loss_i, new_mems_i, grads_and_vars_i = single_core_graph(
            n_token=n_token,
            cutoffs=cutoffs,
            is_training=True,
            inp=inputs[i],
            tgt=labels[i],
            mems=mems_i,
            conf=conf)

        tower_mems.append(mems_i)
        tower_losses.append(loss_i)
        tower_new_mems.append(new_mems_i)
        tower_grads_and_vars.append(grads_and_vars_i)

    ## average losses and gradients across towers
    if len(tower_losses) > 1:
      loss = tf.add_n(tower_losses) / len(tower_losses)
      grads_and_vars = average_grads_and_vars(tower_grads_and_vars)
    else:
      loss = tower_losses[0]
      grads_and_vars = tower_grads_and_vars[0]
    grads, all_vars = zip(*grads_and_vars)

    ## clip gradient
    clipped, gnorm = tf.clip_by_global_norm(grads, conf['clip'])
    grads_and_vars = list(zip(clipped, all_vars))

    ## configure the optimizer
    global_step = tf.train.get_or_create_global_step()

    # warmup stage: increase the learning rate linearly
    if conf['warmup_steps'] > 0:
      warmup_lr = tf.to_float(global_step) / tf.to_float(conf['warmup_steps']) \
                  * conf['learning_rate']
    else:
      warmup_lr = 0.0

    # decay stage: decay the learning rate using the cosine schedule
    decay_lr = tf.train.cosine_decay(
        conf['learning_rate'],
        global_step=global_step-conf['warmup_steps'],
        decay_steps=conf['train_steps']-conf['warmup_steps'],
        alpha=conf['min_lr_ratio'])

    # choose warmup or decay
    learning_rate = tf.where(global_step < conf['warmup_steps'],
                             warmup_lr, decay_lr)

    # get the train op
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step)

    ##### Training loop
    tower_mems_np = [
        [np.zeros([conf['mem_len'], per_core_bsz, conf['d_model']], dtype=np.float32)
            for layer in range(conf['n_layer'])]
        for core in range(conf['num_core_per_host'])
    ]

    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      sess.run(tf.global_variables_initializer())

      if conf['warm_start_path'] is not None:
        tf.logging.info("warm start from {}".format(conf['warm_start_path']))
        saver.restore(sess, conf['warm_start_path'])

      fetches = [loss, tower_new_mems, global_step, gnorm, learning_rate, train_op]

      total_loss, prev_step = 0., -1
      while True:
        feed_dict = {}
        for i in range(conf['num_core_per_host']):
          for m, m_np in zip(tower_mems[i], tower_mems_np[i]):
            feed_dict[m] = m_np

        fetched = sess.run(fetches, feed_dict=feed_dict)

        loss_np, tower_mems_np, curr_step = fetched[:3]
        total_loss += loss_np

        if curr_step > 0 and curr_step % conf['iterations'] == 0:
          curr_loss = total_loss / (curr_step - prev_step)
          tf.logging.info("[{}] | gnorm {:.2f} lr {:8.6f} "
              "| loss {:.2f} | pplx {:>7.2f}, bpc {:>7.4f}".format(
              curr_step, fetched[-3], fetched[-2],
              curr_loss, math.exp(curr_loss), curr_loss / math.log(2)))
          total_loss, prev_step = 0., curr_step

        if curr_step > 0 and curr_step % conf['save_steps'] == 0:
          save_path = os.path.join(conf['model_dir'], "model.ckpt")
          saver.save(sess, save_path)
          tf.logging.info("Model saved in path: {}".format(save_path))

        if curr_step == conf['train_steps']:
          break


def evaluate(n_token, cutoffs, ps_device, conf):
    ##### Get input function and model function
    eval_input_fn, eval_record_info = bumblebeat.data.get_input_fn(
        record_info_dir=conf['record_info_dir'],
        split=conf['eval_split'],
        per_host_bsz=conf['eval_batch_size'],
        tgt_len=conf['tgt_len'],
        num_core_per_host=conf['num_core_per_host'],
        num_hosts=1,
        use_tpu=False)

    num_batch = eval_record_info["num_batch"]
    if conf['max_eval_batch'] > 0:
        num_batch = conf['max_eval_batch']
    tf.logging.info("num of batches {}".format(num_batch))

    ##### Create computational graph
    eval_set = eval_input_fn({
        "batch_size": conf['eval_batch_size'],
        "data_dir": conf['data_dir']})

    input_feed, label_feed = eval_set.make_one_shot_iterator().get_next()

    inputs = tf.split(input_feed, conf['num_core_per_host'], 0)
    labels = tf.split(label_feed, conf['num_core_per_host'], 0)

    per_core_bsz = conf['eval_batch_size'] // conf['num_core_per_host']
    tower_mems, tower_losses, tower_new_mems = [], [], []

    for i in range(conf['num_core_per_host']):
      with tf.device(assign_to_gpu(i, ps_device)), \
          tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):

        mems_i = [tf.placeholder(tf.float32,
                      [conf['mem_len'], per_core_bsz, conf['d_model']])
                  for _ in range(conf['n_layer'])]

        loss_i, new_mems_i = single_core_graph(
            n_token=n_token,
            cutoffs=cutoffs,
            is_training=False,
            inp=inputs[i],
            tgt=labels[i],
            mems=mems_i,
            conf=conf)

        tower_mems.append(mems_i)
        tower_losses.append(loss_i)
        tower_new_mems.append(new_mems_i)

    ## sum losses across towers
    if len(tower_losses) > 1:
      loss = tf.add_n(tower_losses) / len(tower_losses)
    else:
      loss = tower_losses[0]

    ##### Evaluation loop
    tower_mems_np = [
        [np.zeros([conf['mem_len'], per_core_bsz, conf['d_model']], dtype=np.float32)
            for layer in range(conf['n_layer'])]
        for core in range(conf['num_core_per_host'])
    ]

    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      sess.run(tf.global_variables_initializer())

      if conf['eval_ckpt_path'] is None:
        eval_ckpt_path = tf.train.latest_checkpoint(conf['model_dir'])
      else:
        eval_ckpt_path = conf['eval_ckpt_path']
      tf.logging.info("Evaluate {}".format(eval_ckpt_path))
      saver.restore(sess, eval_ckpt_path)

      fetches = [loss, tower_new_mems, tf.size(label_feed)]

      format_str = "  >> processing batch {{:{0}d}}/{{:{0}d}} ..".format(
          len(str(num_batch)))

      total_loss, total_cnt = 0, 0
      for step in range(num_batch):
        if step % (num_batch // 10) == 0:
          tf.logging.info(format_str.format(step, num_batch))

        feed_dict = {}
        for i in range(conf['num_core_per_host']):
          for m, m_np in zip(tower_mems[i], tower_mems_np[i]):
            feed_dict[m] = m_np

        fetched = sess.run(fetches, feed_dict=feed_dict)

        loss_np, tower_mems_np, cnt_np = fetched[:3]
        total_loss += loss_np * cnt_np
        total_cnt += cnt_np

      avg_loss = total_loss / total_cnt
      tf.logging.info("| loss {:.2f} | pplx {:>7.2f}, bpc {:>7.4f}".format(
          avg_loss, math.exp(avg_loss), avg_loss / math.log(2)))


def model_main(conf):
    """
    Train model on dataset using parameters specified
    in <conf>. Requires that the data step has ran successfully
    and TF Records are available.
    """
    model_conf = conf['model']
    tf.logging.set_verbosity(tf.logging.INFO)

    # Get corpus info
    corpus_info = bumblebeat.data.get_corpus_info(model_conf['corpus_info_path'])
    n_token = corpus_info["vocab_size"]
    cutoffs = corpus_info["cutoffs"][1:-1]
    tf.logging.info("n_token {}".format(n_token))

    if model_conf['do_train']:
        train(n_token, cutoffs, "/gpu:0", model_conf)
    if model_conf['do_eval']:
        evaluate(n_token, cutoffs, "/gpu:0", model_conf)
