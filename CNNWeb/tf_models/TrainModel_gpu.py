import json
import time
from datetime import datetime

import tensorflow as tf

import CreateModel_gpu as CreateModel
from CNNWeb.tf_models import FileManager

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', './tmp/train', """
event logs + checkpoints
""")
tf.app.flags.DEFINE_integer('max_steps',100000,"number of batchs to run")
tf.app.flags.DEFINE_boolean('log_device_placement',False,"whether to log device placement")
tf.app.flags.DEFINE_integer('log_frequency',10,"how often to log result")
NUM_PER_BATCH = 100
MAX_STEP =10000


with open('settings.json','r') as setting:
    data = json.load(setting)
flist_str = data['DataFile']
BASE_PATH = data['BasePath']
LOCAL_FILE_LIST = str(flist_str).split(',')
FILE_LIST = []
for l in LOCAL_FILE_LIST:
    s = BASE_PATH+l
    FILE_LIST.append(s)
def train():
    with tf.device('/cpu:0'):
        with tf.Graph().as_default():
            global_step = tf.contrib.framework.get_or_create_global_step()
            fm = FileManager.FileManager()
            try:
                logit ,label = fm.read_and_decode(FILE_LIST)
                logits_batch , labels_batch = CreateModel.input(logit,label)
                print(logits_batch)
                print(labels_batch)
            except IOError as e:
                print('File not find : ')
                print(e.errno)
                return
            with tf.device('/cpu:0'):
                logits = CreateModel.interface(logits = logits_batch)
                loss = CreateModel.loss(logits,label=labels_batch)
                #print(tf.global_variables())
                train_op = CreateModel.train(totalloss=loss,global_step=global_step)
            class _loghooker(tf.train.SessionRunHook):
                def begin(self):
                    self._step = -1
                    self._start_time = time.time()
                    pass
                def before_run(self,run_context):
                    self._step+=1
                    return tf.train.SessionRunArgs(loss)
                    pass
                def after_run(self, run_context, run_values):
                    if self._step%FLAGS.log_frequency == 0:
                        current_time = time.time()
                        duration = current_time - self._start_time
                        self._start_time = current_time
                        loss_value = run_values.results
                        example_per_second = FLAGS.log_frequency * FLAGS.batch_size / duration
                        sec_per_batch = float(duration/FLAGS.log_frequency)
                        format_str= ('%s : step %d,loss = %.2f(%.1f examples/sec;%.3f sec/batch)')
                        print(format_str % (datetime.now(), self._step,loss_value,example_per_second,sec_per_batch))


            with tf.train.MonitoredTrainingSession(
                checkpoint_dir=FLAGS.train_dir,
                hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                       tf.train.NanTensorHook(loss),
                       _loghooker()],
                config = tf.ConfigProto(log_device_placement = FLAGS.log_device_placement)
            ) as mon_sess:

                while not mon_sess.should_stop():
                    mon_sess.run(train_op)
def main(argv = None):

    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()
if __name__ == '__main__':
    tf.app.run()