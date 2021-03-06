import numpy as np
import tensorflow as tf
import re
import pandas as pd
import pickle as pk


def next_batch(data, labels, batch_size):
    import random
    idxs = random.sample(range(labels.shape[0]), batch_size)
    return [data[i] for i in idxs], labels[idxs]

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

def read_data(filename, sen_len=400):
    data = pd.read_csv(filename, delimiter='ññ', header=None, engine='python')
    with open('w2v/model/nce_embeddings.pkl','rb') as f:
        emb = pk.load(f)
    with open('w2v/model/nce_dict.pkl','rb') as f:
        w_dict = pk.load(f)
    new_data = []
    for tweet in data[0][:]:
        sen_matrix = []
        for word in preprocess(tweet):
            if word.startswith('@'):
                word = '<USER/>'
            elif word.startswith('http'):
                word = '<URL/>'
            elif word.startswith('#'):
                word = '<HASHTAG/>'
            else:
                word = word.lower()
            if word in w_dict:
                sen_matrix += [ emb[ w_dict[ word ] ] ]
            else:
                sen_matrix += [ emb[ w_dict[ 'UNK' ] ] ]
        ## UNK padding
        missing = sen_len - len(sen_matrix)
        for x in range(missing):
            sen_matrix += [ emb[ w_dict[ 'UNK' ] ] ] ## embeddings of lenght 100 each
        new_data += [sen_matrix]
    return np.array(data),np.array(pd.get_dummies(data[1][:]).as_matrix())

class Cnn:

  def __init__(self, ss,d, m, fm):
    self.embed_size = d   ## size of the embeddings
    self.filter_size = m  ## size of the filter
    self.fm_num = fm      ## feature maps number
    self.sen_siz = ss     ## fixed size of sentence
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
    with tf.name_scope('input'):
      # placeholder for X
      self.X = tf.placeholder(tf.float32, [None, self.embed_size], name='X')
      # placeholder for Y
      self.Y_true = tf.placeholder(tf.float32, [None, 3], name='Y')

  def def_params(self):
    """ Defines model parameters """
    with tf.name_scope('params'):
      # First convolutional layer - maps one grayscale image to 2x32 feature maps.
      with tf.name_scope('conv'):
        self.W_cn = self.weight_variable([5, self.embed_size, 1, self.fm_num])
        self.b_cn = self.bias_variable([self.fm_num])
      with tf.name_scope('fc_softmax'):
        self.W_fc = self.weight_variable([300, 3]) ## por los 300 mapas de características reducidos por max pooling
        self.b_fc = self.bias_variable([3])

  def def_model(self):
    """ Defines the model """
    self.Xm = self.X
    self.W_cnm = self.W_cn
    self.b_cnm = self.b_cn
    self.W_fcm = self.W_fc
    self.b_fcm = self.b_fc
    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('zero_padding'):
      zero_x = tf.pad(self.Xm,[[0,0],[0,self.sen_siz - self.X.shape[0]]])

    with tf.name_scope('conv'):
      h_cn1 = tf.nn.relu(self.conv2d(self.Xm, self.W_cnm) + self.b_cnm)
    # Pooling layer - downsamples by 2X.
    with tf.name_scope('max_pool'):
      h_max_pool = self.max_pool(h_cn1)
    with tf.name_scope('fc_softmax'):
      self.Y_logt = tf.matmul(h_max_pool, self.W_fcm) + self.b_fcm
      self.Y_pred = tf.nn.softmax(self.Y_logt)

  def def_output(self):
    """ Defines model output """
    with tf.name_scope('output'):
      self.label_pred = tf.argmax(self.Y_pred, 1, name='label_pred')
      self.label_true = tf.argmax(self.Y_true, 1, name='label_true')

  def def_loss(self):
    """ Defines loss function """
    with tf.name_scope('loss'):
      # cross entropy
      self.cross_entropy = (tf.nn.softmax_cross_entropy_with_logits(labels=self.Y_true, logits=self.Y_logt) +
                            0.01*tf.nn.l2_loss(self.W_cnm) +
                            0.01*tf.nn.l2_loss(self.b_cnm) +
                            0.01*tf.nn.l2_loss(self.W_fcm) +
                            0.01*tf.nn.l2_loss(self.b_fcm))
      self.loss = tf.reduce_mean(self.cross_entropy)

  def def_metrics(self):
    """ Adds metrics """
    with tf.name_scope('metrics'):
      cmp_labels = tf.equal(self.label_true, self.label_pred)
      self.accuracy = tf.reduce_mean(tf.cast(cmp_labels, tf.float32), name='accuracy')

  def add_summaries(self):
    """ Adds summaries for Tensorboard """
    # defines a namespace for the summaries
    with tf.name_scope('summaries'):
      # adds a plot for the loss
      tf.summary.scalar('loss', self.loss)
      tf.summary.scalar('accuracy', self.accuracy)
      # groups summaries
      self.summary = tf.summary.merge_all()

  def conv2d(self, x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  def max_pool(self, x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.reduce_max(tf.reduce_max(x, axis=2),axis=0)

  def weight_variable(self, shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(self, shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def train(self, data, labels):
    """ Trains the model """
    # creates optimizer
    grad = tf.train.AdadeltaOptimizer(learning_rate=1.0)
    # setup minimize function
    optimizer = grad.minimize(self.loss)

    # opens session
    with tf.Session() as sess:
      # initialize variables (params)
      sess.run(tf.global_variables_initializer())
      # writers for TensorBorad
      train_writer = tf.summary.FileWriter('graphs/sentiment_train')
      test_writer = tf.summary.FileWriter('graphs/sentiment_test')
      train_writer.add_graph(sess.graph)

      #  batches
      cut_idx = int(len(labels)*.2)
      X_test, Y_test = data[cut_idx:], labels[cut_idx:]
      tr_data = data[:cut_idx]
      tr_labels = labels[:cut_idx]
      del data
      del labels
      # training loop
      for i in range(550*3):

        # train batch
        #~ X_train, Y_train = data.train.next_batch(10)
        X_train, Y_train = next_batch( tr_data, tr_labels, 200 )

        #~ # evaluation with train data
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


def run():
  # Tensorflow integrates MNIST dataset
  data, labels = read_data('data/distant-data.ds')
  print("data size: ", len(data))
  # defines our model
  model = Cnn(400, 100, 5, 300)
  # trains our model
  model.train(data, labels)

def main(args):
  run()
  return 0

if __name__ == '__main__':
  import sys
  sys.exit(main(sys.argv))
