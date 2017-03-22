import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import tensorflow as tf
import jieba
from gensim.models import Word2Vec
BASE_PATH= ''
WORD_VEC_FILE = ''
DIC = None
def _get_dic_file():

    try:
        dic = Word2Vec.load(BASE_PATH+WORD_VEC_FILE)
    except ImportError :
        print 'File error'
    return dic
def _generate_vector_and_labels(seg):

    if DIC is None:
        print 'No dic initilized'
    label = seg[0]
    string_vec = []
    for word in seg[1:]:
        word_vec = DIC[word]
        string_vec.append(word_vec)
    return string_vec
def input(textfile):
    """

    :param textfile:string 输入数据文件地址集合
    :return: [vectors，labels]输出的是向量和他对应的标签
    """
    DIC = _get_dic_file()
    for filename in textfile:
        source_file = open(filename,'r')
        for line in source_file:
            seg = str(line).split(' ')
            str_vec = _generate_vector_and_labels(seg)
            yield str_vec
    return