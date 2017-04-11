import tensorflow as tf
from gensim.models import Word2Vec
import sys
import jieba
from imp import reload
import os
import xml.sax
reload(sys)
class FileManager:
    Base_Path = 'D:/Python/op/data/'
    Unknown_Vec = []
    counter = 0
    DIC_path = Base_Path+'Word60.model'
    def __init__(self):
        self._get_dic()
        for i in range(0,60):
            self.Unknown_Vec.append(0.0)
        return
    def _get_dic(self):
        try:
            self.dic = Word2Vec.load(self.DIC_path)
        except IOError as i:
            print('Word Vector not found')
    def generate_TFRecord_file(self,filename):
        writer = tf.python_io.TFRecordWriter(self.Base_Path+'data.tfrecords')
        for file in filename:
            try:
                fopen = open(self.Base_Path+'labeled/'+file,'r',encoding='utf-8')
            except IOError as i :
                print(i)
                print('file \''+file+'\' not found')
                continue
            print ('analysis '+file+':')
            labels = []
            index = 0
            for line in fopen:

                words = jieba.lcut(str(line))
                sys.stdout.write('current word: '+str(self.counter)+'\t\tFile name:'+file+'\r')
                sys.stdout.flush()
                self.counter+=1
                vecs = []
                try:
                    if words[0] == u'\ufeff' or words[0]=='\n':
                        label = words[1]
                        word_alt = words[2:]
                    else:
                        label = words[0]
                        word_alt = words[1:]

                except IndexError:
                    continue
                index += 1
                for i in range(0,140):

                    try:
                        vecs.extend(self.dic[word_alt[i]].tolist())
                    except KeyError :

                        vecs.extend(self.Unknown_Vec)
                        continue
                    except IndexError :
                        vecs.extend(self.Unknown_Vec)
                        continue
                example = tf.train.Example(features = tf.train.Features(feature = {
                    "label" : tf.train.Feature(int64_list = tf.train.Int64List(value = [int(label)])),
                    "vecs"  : tf.train.Feature(float_list = tf.train.FloatList(value=vecs))
                }))
                writer.write(example.SerializeToString())
        writer.close()
    def read_and_decode(self,filename):
        fq = tf.train.string_input_producer(filename)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(fq)
        example = tf.parse_single_example(serialized_example, features={
            'label':tf.FixedLenFeature([],tf.int64),
            'vecs' :tf.VarLenFeature(tf.float32)
        })

        vecs = tf.sparse_reshape(example['vecs'],[140,60,1])

        vecs = tf.sparse_to_dense(vecs.indices,[140,60,1],vecs.values)
        vecs = tf.cast(vecs,tf.float32)
        label = tf.cast(example['label'], tf.int32)
        return vecs, label
def show_dic(fm):
    fm._get_dic()
    print(fm.dic)

class Weibo_handler(xml.sax.ContentHandler):
    Currenttype = ''
    label = 1
    emontion = ''
    txtfile = open('weibo.txt','w',encoding='utf-8')
    content = ''
    emotionlist = []
    positive = ['高兴','喜好','惊讶']
    negative = ['厌恶','恐惧','悲伤','愤怒']
    middle = ['无','D','']
    def startElement(self, name, attrs):

        self.Currenttype = name
        if name == 'sentence'and attrs['opinionated'] == 'Y':
            try:
                # if attrs['emotion-type'] not in self.emotionlist:
                #     self.emotionlist.append(attrs['emotion-type'])
                self.emontion = attrs['emotion-type']
            except KeyError:
                # if attrs['emotion-type1'] not in self.emotionlist:
                #     self.emotionlist.append(attrs['emotion-type1'])
                self.emontion = attrs['emotion-1-type']
        elif name == 'sentence':
            self.emontion = ''
    def characters(self, content):
        if self.Currenttype == 'sentence':
            self.label = 1
            if self.emontion in self.positive:
                self.label = 0
            elif self.emontion in self.negative:
                self.label = 2
            self.content = content
    def endElement(self, name):
        if self.Currenttype == 'sentence':
            self.txtfile.write(str(self.label)+' '+self.content+'\n')
        self.Currenttype =''


def main():
    fm = FileManager()
    filename = os.listdir(fm.Base_Path+'labeled/')

    fm.generate_TFRecord_file(filename)
def readXML():
    fm = FileManager()
    xmlh = Weibo_handler()
    res = xml.sax.parse(fm.Base_Path+'微博情绪标注语料.xml',xmlh)

def main2r():
    fm = FileManager()
    vecs,label = fm.read_and_decode([fm.Base_Path+'data.tfrecords'])
    vecbatch , labelbatch = tf.train.batch([vecs, label],3, 1)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        threads = tf.train.start_queue_runners(sess=sess)
        result = sess.run([vecbatch,labelbatch])
        print(result[0])
if __name__ == '__main__':
    main()