import tensorflow as tf
import numpy as np

import time
import argparse
import os
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--pach_task_id", help="specify pach task 1-20 (default=1)")
parser.add_argument("-r", "--restore", help="specify restore weight directory (default=None)")
parser.add_argument("-s", "--strong_supervision", help="use labelled supporting facts (default=false)")
parser.add_argument("-t", "--dmn_type", help="specify type of dmn (default=original)")
parser.add_argument("-l", "--l2_loss", type=float, default=0.001, help="specify l2 loss constant")
parser.add_argument("-n", "--num_runs", type=int, help="specify the number of model runs")
parser.add_argument("-y", "--yes_repeat", type=int, help="specify the number of repeated yes training inputs")
parser.add_argument("-v", "--vec_size", type=int, help="specify GloVe vec_size, default=100")
parser.add_argument("-m", "--max_inputs", type=int, help="specify max allowed input length")
parser.add_argument("-S", "--batch_size", type=int, help="specify pach task")
parser.add_argument("-V", "--min_vocab_cnt", type=int, help="specify min vocab count")

begin = time.time()
args = parser.parse_args()

#pdb.set_trace()

dmn_type = args.dmn_type if args.dmn_type is not None else "plus"

if dmn_type == "original":
    from dmn_original import Config
    config = Config()
elif dmn_type == "plus":
    from dmn_plus import Config
    config = Config()
else:
    raise NotImplementedError(dmn_type + ' DMN type is not currently implemented')

if args.pach_task_id is not None:
    config.pach_id = args.pach_task_id

config.pach_id = args.pach_task_id if args.pach_task_id is not None else str(1)
config.l2 = args.l2_loss if args.l2_loss is not None else 0.001
config.strong_supervision = args.strong_supervision if args.strong_supervision is not None else False
num_runs = args.num_runs if args.num_runs is not None else 1
config.yes_repeat = args.yes_repeat if args.yes_repeat is not None else 1
config.vec_size = args.vec_size if args.vec_size is not None else 100
config.hidden_size = config.embed_size = config.vec_size
config.max_allowed_inputs = args.max_inputs if args.max_inputs is not None else config.max_allowed_inputs
config.batch_size = args.batch_size if args.batch_size is not None else config.batch_size
config.min_vocab_cnt = args.min_vocab_cnt if args.min_vocab_cnt is not None else config.min_vocab_cnt

print 'Training DMN ' + dmn_type + ' on pach task', config.pach_id
print "max_input_len="+str(config.max_allowed_inputs)
print "max_allowed_sent_len="+str(config.max_allowed_sent_len)
print "min_vocab_cnt ="+str(config.min_vocab_cnt)
print "episodic memory num_hops="+str(config.num_hops)
print "embed_size ="+str(config.embed_size) # 100
print "hidden_size ="+str(config.hidden_size) # 100

best_overall_val_loss = float('inf')

# create model
with tf.variable_scope('DMN') as scope:
    if dmn_type == "original":
        from dmn_original import DMN
        model = DMN(config)
    elif dmn_type == "plus":
        from dmn_plus import DMN_PLUS
        model = DMN_PLUS(config)

for run in range(num_runs):

    print 'Starting run', run

    print '==> initializing variables'
    init = tf.global_variables_initializer()
    #init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    with tf.Session() as session:

        sum_dir = 'summaries/train/' + time.strftime("%Y-%m-%d %H %M")
        if not os.path.exists(sum_dir):
            os.makedirs(sum_dir)
        train_writer = tf.summary.FileWriter(sum_dir, session.graph)
        #train_writer = tf.train.SummaryWriter(sum_dir, session.graph)

        session.run(init)

        best_val_epoch = 0
        prev_epoch_loss = float('inf')
        best_val_loss = float('inf')
        best_val_accuracy = 0.0

        if args.restore:
           # pdb.set_trace()
           # print '==> restoring weights'
           # saver.restore(session, 'weights/task' + str(model.config.pach_id) + '.weights')
            weight_dir = args.restore
            print '==> restoring weights from '+weight_dir+'/task' + str(model.config.pach_id) + '.weights'
            saver.restore(session, weight_dir+'/task' + str(model.config.pach_id) + '.weights')

        print '==> starting training'
        for epoch in xrange(config.max_epochs):
            print 'Epoch {}'.format(epoch)
            start = time.time()
            #pdb.set_trace()
            train_loss, train_accuracy, pred, ans, p = model.run_epoch(
              session, model.train, epoch, train_writer,
              train_op=model.train_step, train=True)
            valid_loss, valid_accuracy, vpred, vans, vp = model.run_epoch(session, model.valid)
            #pdb.set_trace()
            print 'Training loss: {}'.format(train_loss)
            print 'Validation loss: {}'.format(valid_loss)
            print 'Training accuracy: {}'.format(train_accuracy)
            print 'Vaildation accuracy: {}'.format(valid_accuracy)

            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                best_val_epoch = epoch
                if best_val_loss < best_overall_val_loss:
                    print 'Saving weights'
                    best_overall_val_loss = best_val_loss
                    best_val_accuracy = valid_accuracy
                    saver.save(session, 'weights/task' + str(model.config.pach_id) + '.weights')

            # anneal
            if train_loss>prev_epoch_loss*model.config.anneal_threshold:
                model.config.lr/=model.config.anneal_by
                print 'annealed lr to %f'%model.config.lr

            prev_epoch_loss = train_loss

            if epoch - best_val_epoch > config.early_stopping:
                break
            print 'Total time: {}'.format(time.time() - start)
        print 'Best validation accuracy:', best_val_accuracy


print 'Total Train time: {}'.format(time.time() - begin)

