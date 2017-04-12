import json
import FileManager as FM
import CreateModel
tf  = FM.tf
SETTINGS = json.load(open('setting.json','r'))
FILE = SETTINGS['EvaluateFile']
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir','./tmp/eval','Dictiory to store event log')
tf.app.flags.DEFINE_string('eval_data','test',"File ")
tf.app.flags.DEFINE_string('checkpoint_dir','./tmp/train',"Direction to store train imformation")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")

def eval_once(saver):

    with tf.Session() as session:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session,ckpt.model_checkpoint_path)
        else:
            print("no ckpt file found")
        coord = tf.train.Coordinator()
        try:
            thread = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                print qr
                thread.extend(qr.create_threads(session,coord=coord,daemon=True,start=True))
        except Exception as e:
            print (e)
        finally:

            coord.request_stop(e)




def evaluate():
    with tf.Graph.as_default():
        fm = FM.FileManager()
        vecs,label = fm.read_and_decode(FILE)
        logits = CreateModel.interface(logits=vecs)
        top_k = tf.nn.in_top_k(logits,label,1)

        ema = tf.train.ExponentialMovingAverage(CreateModel.DECAY_RATE)
        variabel_to_restore = ema.variables_to_restore()
        saver = tf.train.Saver(variabel_to_restore)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir,g)
        while True:
            eval_once(saver)
eval_once(None)