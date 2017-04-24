import json
import time
from datetime import datetime
import os
import tensorflow as tf

from CNNWeb.tf_models import FileManager, Create

class Train():

    flist_str = ''
    BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/data/'
    Record_Path = ''
    LOCAL_FILE_LIST = ''
    FILE_LIST = []
    BATCH_SIZE = 0
    TRAIN_DIR = './tmp/train/'
    MAX_STEPS = 0
    LOG_FREQUENCY = 10
    settings = None
    def __init__(self,**args):
        self.settings = args
        with open('settings.json', 'r') as setting:
            data = json.load(setting)
        date = str(args['date_time']).split('.')[0]
        self.TRAIN_DIR += args['name']+data
        self.flist_str = args['data_set']
        self.Record_Path = self.BASE_PATH + data['RecordPath']
        self.LOCAL_FILE_LIST = str(self.flist_str).split(',')
        self.BATCH_SIZE = args['batch_size']
        self.MAX_STEPS = args['max_step']
        for l in self.LOCAL_FILE_LIST:
            s = self.Record_Path+l
            self.FILE_LIST.append(s)
    def generate_record(self):
        fm = FileManager.FileManager()
        fm.generate_TFRecord_file()
    def train(self):
        with tf.Graph().as_default():
            global_step = tf.contrib.framework.get_or_create_global_step()
            fm = FileManager.FileManager()
            interface = Create.Interface()
            try:
                logit ,label = fm.read_and_decode(self.FILE_LIST)
                logits_batch , labels_batch = interface.input(logit, label)

            except IOError as e:
                print('File not find : ')
                print(e.errno)
                return

            logits = interface.interface(logits = logits_batch)
            loss = interface.loss(logits, label=labels_batch)
            zero_label = tf.zeros([self.BATCH_SIZE],tf.int32)
            p = tf.nn.in_top_k(logits,labels_batch,1)
            z_ls = tf.nn.in_top_k(logits,zero_label,1)
            p_int = tf.cast(p,tf.int32)
            z_int = tf.cast(z_ls,tf.int32)
            ap = tf.reduce_sum(p_int)
            zn = tf.reduce_sum(z_int)

            with tf.control_dependencies([ap,zn]):
            #print(tf.global_variables())
                train_op = interface.train(totalloss=loss, global_step=global_step)
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
                    if self._step%Train.LOG_FREQUENCY == 0:
                        current_time = time.time()
                        duration = current_time - self._start_time
                        self._start_time = current_time
                        loss_value = run_values.results

                        example_per_second = Train.LOG_FREQUENCY * Train.BATCH_SIZE / duration
                        sec_per_batch = float(duration/Train.LOG_FREQUENCY)
                        format_str= ('%s : step %d,loss = %.2f(%.1f examples/sec;%.3f sec/batch)')
                        print(format_str % (datetime.now(), self._step,loss_value,example_per_second,sec_per_batch))

            class _loghooker_zeros(tf.train.SessionRunHook):
                def begin(self):
                    self._step = -1
                    self.all = 0
                    self._start_time = time.time()
                    pass

                def before_run(self, run_context):
                    self._step += 1
                    return tf.train.SessionRunArgs(zn)
                    pass

                def after_run(self, run_context, run_values):
                    zs = run_values.results
                    self.all += zs
                    if self._step%Train.LOG_FREQUENCY==0:

                        print('zeros:'+ str(self.all/Train.BATCH_SIZE/Train.LOG_FREQUENCY))
                        self.all = 0

            class _loghooker_pres(tf.train.SessionRunHook):
                def begin(self):
                    self._all_true = 0
                    self._step = -1
                    self._start_time = time.time()
                    pass
                def before_run(self,run_context):
                    self._step+=1
                    return tf.train.SessionRunArgs(ap)
                    pass
                def after_run(self, run_context, run_values):
                    true_num = run_values.results
                    self._all_true +=true_num

                    if self._step%Train.LOG_FREQUENCY == 0:

                        print(self._all_true/Train.LOG_FREQUENCY/Train.BATCH_SIZE)
                        self._all_true = 0
            with tf.train.MonitoredTrainingSession(
                checkpoint_dir=self.TRAIN_DIR,
                hooks=[tf.train.StopAtStepHook(last_step=Train.MAX_STEPS),
                       tf.train.NanTensorHook(loss),
                       _loghooker(),
                       _loghooker_pres(),
                       _loghooker_zeros()],
                config = tf.ConfigProto(log_device_placement = False)
            ) as mon_sess:

                while not mon_sess.should_stop():
                    mon_sess.run(train_op)
def run(**argv):
    train = Train(argv)
    if tf.gfile.Exists(train.TRAIN_DIR):
        tf.gfile.DeleteRecursively(train.TRAIN_DIR)
    tf.gfile.MakeDirs(train.TRAIN_DIR)
    train.train()
if __name__ == '__main__':

    tf.app.run()