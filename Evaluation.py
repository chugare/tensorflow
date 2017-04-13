import json
import FileManager as FM
import Create
import math
import numpy

tf = FM.tf
SETTINGS = json.load(open('settings.json', 'r'))
BASE_PATH = SETTINGS['BasePath']
FILE = SETTINGS['EvaluateFile']
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', './tmp/eval', 'Dictiory to store event log')
tf.app.flags.DEFINE_string('eval_data', 'test', "File ")
tf.app.flags.DEFINE_string('checkpoint_dir', './tmp/train', "Direction to store train imformation")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 1000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")


def eval_once(saver, summary_op, top_k_op, summary_writer):
    with tf.Session() as session:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print("no ckpt file found")
        coord = tf.train.Coordinator()
        try:
            thread = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                print(qr)
                thread.extend(qr.create_threads(session, coord=coord, daemon=True, start=True))
            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            true_count = 0
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            print(coord.should_stop())
            while step < num_iter and not coord.should_stop():
                predictions = session.run([top_k_op])
                true_count += numpy.sum(predictions)
                step += 1
                print("%d steps , precision: %.3f " % (step, true_count / (step*FLAGS.batch_size)))

            res = true_count / total_sample_count
            print('precision@1: %.3f' % res)
        except Exception as e:
            print(e)
            pass
        finally:
            coord.request_stop()
        coord.join(thread, 10)


def batch_evaluate():
    with tf.Graph().as_default() as g:
        fm = FM.FileManager()
        vecs, label = fm.read_and_decode([BASE_PATH + FILE])
        vecs, label = Create.input(vecs, label)
        logits = Create.interface(logits=vecs)
        top_k = tf.nn.in_top_k(logits, label, 1)

        ema = tf.train.ExponentialMovingAverage(Create.DECAY_RATE)
        variabel_to_restore = ema.variables_to_restore()
        saver = tf.train.Saver(variabel_to_restore)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)
        while True:
            eval_once(saver, summary_op, top_k, summary_writer)

def single_evaluate():
    with tf.Graph().as_default() as g:
        vecs_pl = tf.placeholder(tf.float32,[140,60])
        vecs_batch = tf.reshape(vecs_pl,[1,140,60,1])
        #vecs_batch = tf.train.shuffle_batch([vecs],1,1000,10)
        prediction_res = Create.interface(vecs_batch)

        ema = tf.train.ExponentialMovingAverage(Create.DECAY_RATE)
        variabel_to_restore = ema.variables_to_restore()
        saver = tf.train.Saver(variabel_to_restore)
        fm = FM.FileManager()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("no ckpt file found")
            endF = False

            while not endF:
                sentence = input()
                if str(sentence).startswith("exit()"):
                    break
                vecs_raw = fm.vecs_generte(sentence)
                res_run = sess.run(prediction_res,feed_dict={vecs_pl:vecs_raw})
                print(res_run)
single_evaluate()