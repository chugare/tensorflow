import tensorflow as tf
from gensim.models import Word2Vec
import sys
import jieba
from imp import reload
reload(sys)
print(u'\ufeff')
class FileManager:
    Base_Path = 'D:/Python/op/data/'
    Unknown_Vec = []
    DIC_path = Base_Path+'Word60.model'
    def __init__(self):
        self._get_dic()
        for i in range(0,60):
            self.Unknown_Vec.append(0.0)
        return
    def _get_dic(self):
        try:
            self.dic = Word2Vec.load(self.DIC_path)
        except IOError:
            print('Word Vector not found')
    def generate_TFRecord_file(self,filename):
        writer = tf.python_io.TFRecordWriter(self.Base_Path+'data.tfrecords')
        for file in filename:
            try:
                fopen = open(self.Base_Path+file,'r',encoding='utf-8')
            except IOError:
                print('file \''+file+'\' not found')
                continue
            print ('analysis '+file+':')
            labels = []
            index = 0
            for line in fopen:

                words = jieba.lcut(str(line))
                print('current line: '+str(index))
                print(words)
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
                print('length:'+str(len(vecs)))
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


def main():
    fm = FileManager()
    filename = []
    for i in range(0,2):
        try :
            filename.append(str(i)+'.txt')
        except IOError:
            print('file "'+str(i)+'.txt" not exist')
    fm.generate_TFRecord_file(filename)


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