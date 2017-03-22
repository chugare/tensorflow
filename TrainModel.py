import tensorflow as tf
import CreateModel
import Sentence2Vec
NUM_PER_BATCH = 100
MAX_STEP =10000
FILE_LIST =[]
def train():
    vec_input = Sentence2Vec.input
    interface = CreateModel.interface
    loss = CreateModel.loss
    train_op = CreateModel.train
    str_input = vec_input(FILE_LIST)
    while vec_input()