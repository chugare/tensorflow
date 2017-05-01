import json
import math
import os
import numpy

from tf_models import FileManager as FM
from . import Create

tf = FM.tf
class Eval_Pro():
    SETTINGS = json.load(open('settings.json', 'r'))
    BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/data/'
    Record_path = BASE_PATH+SETTINGS['RecordPath']
    FILE = SETTINGS['EvaluateFile']
    FLAGS = tf.app.flags.FLAGS
    eval_dir = '/tmp/eval'
    checkpoint_dir = './tmp/train'
    eval_interval_secs = 60*5
    num_examples = 1000

    batch_size = 1
    run_thread= None
    def __init__(self,model,thread):
        self.model = model
        self.batch_size = model.batch_size
        self.run_thread = thread
        self.FILE = str(model.id)+'e.tfrecords'
        self.checkpoint_dir+='/'+str(model.id)
        self.ema = model.ema
    def eval_once(self,saver, summary_op, top_k_op, summary_writer):
        with tf.Session() as session:
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
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
                num_iter = int(math.ceil(self.num_examples / self.batch_size))
                true_count = 0.0
                total_sample_count = num_iter * self.batch_size
                step = 0
                while step < num_iter and not coord.should_stop():
                    predictions = session.run([top_k_op])
                    true_count += numpy.sum(predictions)
                    step += 1
                    res = true_count / (step * self.batch_size)

                    self.run_thread.change_state('evaluating',step,res)
                    #print("%d steps , precision: %.3f " % (step,res ))

                res_final = true_count / total_sample_count
                print('precision@1: %.3f' % res)
            except Exception as e:
                print(e)
                pass
            finally:
                coord.request_stop()
            coord.join(thread, 10)


    def batch_evaluate(self,filename):
        with tf.Graph().as_default() as g:
            fm = FM.FileManager(self.run_thread)
            #fm.generate_TFRecord_file([filename],True,self.FILE)
            vecs, label = fm.read_and_decode(self.Record_path + self.FILE)
            it = Create.Interface()
            it.custom_args(self.model)
            vecs, label = it.input(vecs, label)
            logits =it.interface(logits=vecs)
            top_k = tf.nn.in_top_k(logits, label, 1)
            ema = tf.train.ExponentialMovingAverage(self.ema)
            variabel_to_restore = ema.variables_to_restore()
            saver = tf.train.Saver(variabel_to_restore)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.eval_dir)
            for _ in range(0,10):
                self.eval_once(saver, summary_op, top_k, summary_writer)

    def single_evaluate(self):
        with tf.Graph().as_default() as g:
            self._vecs_pl = tf.placeholder(tf.float32,[140,60])
            vecs_batch = tf.reshape(self._vecs_pl,[1,140,60,1])
            #vecs_batch = tf.train.shuffle_batch([vecs],1,1000,10)
            it = Create.Interface()
            it.custom_args(self.model)
            self.prediction_res = it.interface(vecs_batch)

            ema = tf.train.ExponentialMovingAverage(self.ema)
            variabel_to_restore = ema.variables_to_restore()
            saver = tf.train.Saver(variabel_to_restore)
            self.fm = FM.FileManager(self.run_thread)
            self.eval_session =  tf.Session()
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(self.eval_session, ckpt.model_checkpoint_path)
            else:
                print("no ckpt file found")
            endF = False


    def run_single_eval(self,sentence):
        if str(sentence).startswith("exit()"):
            return
        vecs_raw = self.fm.vecs_generte(sentence)
        res_run = self.eval_session.run(self.prediction_res, feed_dict={self._vecs_pl: vecs_raw})
        return res_run
# single_evaluate()
# batch_evaluate()

if __name__ == '__main__':
    pass