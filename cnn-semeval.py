import numpy as np
import tensorflow as tf
import re
import pandas as pd
import pickle as pk
import csv
from pathlib import Path

data_file = open('data/temp_semeval_mat.pkl','rb')


def next_batch(batch_size):
    global data_file
    x = []
    y_true = []
    for i in range(batch_size):
        elem = pk.load(data_file)
        x += elem[0]
        if elem[1][0].startswith('po'):
            y_true += [[1,0,0]]
        else:
            y_true += [[0,0,1]]
    return x,y_true

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]

tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)

def tokenize(s):
    return tokens_re.findall(s)

def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

def read_data(filename, outfile, sen_len=400):
    reader = open(filename)
    with open('w2v/model/nce_embeddings.pkl','rb') as f:
      emb = pk.load(f)
    with open('w2v/model/nce_dict.pkl','rb') as f:
      w_dict = pk.load(f)
    vars = open(outfile,'wb')
    for elem in reader:
      text,label = elem.split('ññ')
      sen_matrix = []
      for word in preprocess(text):
        if word.startswith('@'):
          word = '<USER/>'
        elif word.startswith('http'):
          word = '<URL/>'
        elif word.startswith('#'):
          word = '<HASHTAG/>'
        else:
          word = word.lower()
        if word in w_dict:
          sen_matrix += [ [x] for x in emb[ w_dict[ word ] ] ]
        else:
          sen_matrix += [ [x] for x in emb[ w_dict[ 'UNK' ] ] ]
      missing = sen_len - len(sen_matrix)
      for x in range(missing):
        sen_matrix += [ [x] for x in emb[ w_dict[ 'UNK' ] ] ] ## embeddings of lenght 100 each
      pk.dump([[sen_matrix],[label.strip()]], vars)

class Cnn:

  def __init__(self, ss,d, m, fm, b_size):
    self.embed_size = d   ## size of the embeddings
    self.filter_size = m  ## size of the filter
    self.fm_num = fm      ## feature maps number
    self.sen_siz = ss     ## fixed size of sentence
    self.b_size = b_size  ## batch size
    """ Creates the model """
    self.def_input()
    self.def_params()
    self.def_model()
    self.def_output()
    self.def_loss()
    self.def_metrics()
    self.add_summaries()
  def def_input(self):
    """ Defines inputs """
    with tf.name_scope('input2'):
      # placeholder for X
      self.X = tf.placeholder(tf.float32, [self.b_size,self.sen_siz, self.embed_size], name='X')
      # placeholder for Y
      self.Y_true = tf.placeholder(tf.float32, [self.b_size, 3], name='Y')
      self.l2_loss = tf.constant(0.0)

  def def_params(self):
    """ Defines model parameters """
    with tf.name_scope('params2'):
      with tf.name_scope('fc_softmax'):
        self.W_fc = self.weight_variable([self.fm_num, 3], v_name="wfc2") ## por los 300 mapas de características reducidos por max pooling
        self.b_fc = self.bias_variable([3], v_name="bfc2")

  def def_model(self):
    """ Defines the model """
    Xm = self.X
    W_fcm = self.W_fc
    b_fcm = self.b_fc
    with tf.name_scope('reshaping2'):
      x_re = tf.reshape(Xm,[self.b_size,self.sen_siz,self.embed_size,1])

############################################################################
    num_filters = 100
    pooled_outputs = []
    filter_sizes = [3,5,7]
    for i, filter_size in enumerate(filter_sizes): ## filtros con diferente tamaño
      with tf.name_scope("conv2-maxpool-%s" % filter_size):
        # Convolution Layer
        filter_shape = [filter_size, self.embed_size, 1, num_filters]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
        conv = tf.nn.conv2d( x_re,
              W,
              strides=[1, 1, 1, 1],
              padding="VALID",
              name="conv")

        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

        pooled = tf.nn.avg_pool(
              h,
              ksize=[1, self.sen_siz - filter_size + 1, 1, 1],
              strides=[1, 1, 1, 1],
              padding='VALID',
              name="pool")

        pooled_outputs.append(pooled)
    # Combine all the pooled features
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs,3, name="concat2")
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

############################################################################


    with tf.name_scope('dropout2'):
      h_drop = tf.nn.dropout(h_pool_flat, .4) ## probabilidad de dropout
    with tf.name_scope('fc_softmax2'):
      self.Y_logt = tf.nn.xw_plus_b(h_drop, W_fcm, b_fcm, name='scores')
      self.Y_pred = tf.nn.softmax(self.Y_logt)

  def def_output(self):
    """ Defines model output """
    with tf.name_scope('output2'):
      self.label_pred = tf.argmax(self.Y_pred, 1, name='label_pred')
      self.label_true = tf.argmax(self.Y_true, 1, name='label_true')

  def def_loss(self):
    """ Defines loss function """
    with tf.name_scope('loss2'):
      # cross entropy
      self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.Y_logt,labels=self.Y_true)
      ### lambda de penalización = 0.0001
      self.loss = tf.reduce_mean(self.cross_entropy)

  def def_metrics(self):
    """ Adds metrics """
    with tf.name_scope('metrics2'):
      cmp_labels = tf.equal(self.label_true, self.label_pred)
      self.accuracy = tf.reduce_mean(tf.cast(cmp_labels, tf.float32), name='accuracy')

  def add_summaries(self):
    """ Adds summaries for Tensorboard """
    # defines a namespace for the summaries
    with tf.name_scope('summaries2'):
      # adds a plot for the loss
      tf.summary.scalar('loss', self.loss)
      tf.summary.scalar('accuracy', self.accuracy)
      # groups summaries
      self.summary = tf.summary.merge_all()

  def conv2d(self, x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

  def max_pool(self, x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1,self.sen_siz-5+1,1,1],
                        strides = [1,1,1,1],
                        padding='VALID')

  def weight_variable(self, shape, v_name):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=v_name)

  def bias_variable(self, shape, v_name):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=v_name)

  def train(self):
    """ Trains the model """
    # creates optimizer
    grad = tf.train.AdamOptimizer(learning_rate=.95)
    # setup minimize function
    optimizer = grad.minimize(self.loss)

    # opens session
    with tf.Session() as sess:
      # initialize variables (params)
      sess.run(tf.global_variables_initializer())
      # restore model
      saver = tf.train.import_meta_graph('pre-model/pre-trained-model.meta')
      saver.restore(sess,tf.train.latest_checkpoint('pre-model/'))
      # writers for TensorBorad
      train_writer = tf.summary.FileWriter('graphs-semeval/sentiment_train')
      test_writer = tf.summary.FileWriter('graphs-semeval/sentiment_test')
      train_writer.add_graph(sess.graph)
      saver = tf.train.Saver()

      # training loop
      for i in range(788):

        # train batch
        X_train, Y_train = next_batch(100)
        X_test, Y_test = next_batch(100)

        # evaluation with train data
        feed_dict = {self.X: X_train, self.Y_true: Y_train}
        fetches = [self.loss, self.accuracy, self.summary]
        train_loss, train_acc, train_summary = sess.run(fetches, feed_dict=feed_dict)
        train_writer.add_summary(train_summary, i)

        #~ print(i, train_loss, train_acc)

        #~ # evaluation with test data
        feed_dict = {self.X: X_test, self.Y_true: Y_test}
        fetches = [self.loss, self.accuracy, self.summary]
        test_loss, test_acc, test_summary = sess.run(fetches, feed_dict=feed_dict)
        test_writer.add_summary(test_summary, i)

        # train step
        feed_dict = {self.X: X_train, self.Y_true: Y_train}
        fetches = [optimizer]
        sess.run(fetches, feed_dict=feed_dict)

        # console output
        msg = "I{:3d} loss: ({:6.2f}, {:6.2f}), acc: ({:6.2f}, {:6.2f})"
        msg = msg.format(i, train_loss, test_loss, train_acc, test_acc)
        print(msg)
        if i%100==0:
          saver.save(sess, 'sem-model/semeval-model')



def run():
  # Tensorflow integrates MNIST dataset
  #print("reading data...")
#  if not Path("data/temp_dist_mat.pkl").exists():
 #   read_data(filename='data/distant-data.ds', sen_len=400, outfile='data/temp_dist_mat.pkl')

  # defines our model
  print("instantiating the model...")
  model = Cnn(400, 100, 5, 300, 100)
  # trains our model
  print("training the model...")
  model.train()

def main(args):
  run()

if __name__ == '__main__':
  import sys
  main(sys.argv)
  data_file.close()
  sys.exit(0)
