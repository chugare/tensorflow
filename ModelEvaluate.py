import json
import FileManager as FM
tf  = FM.tf
SETTINGS = json.load(open('setting.json','r'))
FILE = SETTINGS['EvaluateFile']
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir','./tmp/eval','Dictiory to store event log')
tf.app.flags.DEFINE_string('eval_data','')


ef evaluate():
