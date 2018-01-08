import tensorflow as tf
import utils
from tensorflow.python.training import moving_averages
from tensorflow.contrib import rnn

FLAGS = utils.FLAGS
num_classes = utils.num_classes
RNNCell = rnn.RNNCell
from tensorflow.python.util import nest
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import math_ops, array_ops

class RHNCell(RNNCell):

    def __init__(self, num_units, depth=3, forget_bias=None,reuse=None):
        super(RHNCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self.depth = depth
        self.forget_bias = forget_bias

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        current_state = state
        for i in range(self.depth):
            with tf.variable_scope('h_'+str(i)):
                if i == 0:
                    h = tf.tanh(linear([inputs , current_state ], self._num_units, True))
                else:
                    h = tf.tanh(linear([current_state], self._num_units, True))
            with tf.variable_scope('t_'+str(i)):
                if i == 0:
                    t = tf.sigmoid(linear([inputs , current_state], self._num_units, True, self.forget_bias))
                else:         
                    t = tf.sigmoid(linear([current_state], self._num_units, True, self.forget_bias))
            current_state = (h - current_state)* t + current_state

        return current_state, current_state

def linear(args, output_size, bias, bias_start=None, scope=None):
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

# Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    dtype = [a.dtype for a in args][0]

# Now the computation.
    with vs.variable_scope(scope or "Linear"):
        matrix = vs.get_variable(
            "Matrix", [total_arg_size, output_size], dtype=dtype)
        if len(args) == 1:
            res = math_ops.matmul(args[0], matrix)
        else:
            res = math_ops.matmul(array_ops.concat(args, 1), matrix)
        if not bias:
            return res
        elif bias_start is None:
            bias_term = vs.get_variable("Bias", [output_size], dtype=dtype)
        else:
            bias_term = vs.get_variable("Bias", [output_size], dtype=dtype,
                                      initializer=tf.constant_initializer(bias_start, dtype=dtype))
    return res + bias_term

class LSTMOCR(object):
    def __init__(self, mode):
        self.mode = mode
        # image
        self.inputs = tf.placeholder(tf.float32, [None, FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
        # SparseTensor required by ctc_loss op
        self.labels = tf.sparse_placeholder(tf.int32)
        # 1d array of size [batch_size]
        self.seq_len = tf.placeholder(tf.int32, [None])
        # l2
        self._extra_train_ops = []

    def build_graph(self):
        self._build_model()
        self._build_train_op()

        self.merged_summay = tf.summary.merge_all()

    def _build_model(self):
        filters = [64, 128, 256, 512]
        strides = [1, 2]

        with tf.variable_scope('cnn'):
            with tf.variable_scope('unit-1'):
                x = self._conv2d(self.inputs, 'cnn-1', [3,3], 1, filters[0], [1,1])
                #x = self._batch_norm('bn1', x)
                x = self._leaky_relu(x, 0.01)
                x = self._max_pool(x, [2,2], [2,2])
                print(x.shape)

            with tf.variable_scope('unit-2'):
                x = self._conv2d(x, 'cnn-2', [3,3], filters[0], filters[1], [1,1])
                #x = self._batch_norm('bn2', x)
                x = self._leaky_relu(x, 0.01)
                x = self._max_pool(x, [2,2], [2,2])
                print(x.shape)

            with tf.variable_scope('unit-3'):
                x = self._conv2d(x, 'cnn-3_1', [3,3], filters[1], filters[2], [1,1])
                x = self._batch_norm('bn3_1', x)
                x = self._leaky_relu(x, 0.01)

                
                x = self._conv2d(x, 'cnn-3_2', [3,3], filters[2], filters[2], [1,1])
                x = self._leaky_relu(x, 0.01)
                x =self._max_pool(x,[2,1],[2,1])
                  
                print(x.shape)

            with tf.variable_scope('unit-4'):
                x = self._conv2d(x, 'cnn-4_1', [3,3], filters[2], filters[3], [1,1])
                x = tf.contrib.layers.batch_norm(x, scale=True, decay=0.99, scope='bn_41')
                #x = self._batch_norm('bn4_1', x)
                x = self._leaky_relu(x, 0.01)
                x = self._max_pool(x, [2,1], [2,1])
                
                x = self._conv2d(x, 'cnn-4_2', [2,2], filters[3], filters[3], [1,1])
                x = tf.contrib.layers.batch_norm(x, scale=True, decay=0.99, scope='bn_42')
                x = self._batch_norm('bn4_2', x)
                x = self._leaky_relu(x, 0.01)
            
                print(x.shape)
                
            # [batch_size, max_stepsize, num_features]
            
            
                #x = tf.reshape(x, [FLAGS.batch_size,-1,512])
                x = tf.transpose(x, [0, 2, 1,3])
                x = tf.reshape(x, [FLAGS.batch_size,45,-1])
                x.set_shape([FLAGS.batch_size,45, 1024])
                print(x.shape)
                
                    #*************************layer rnn **************************

        for i in range(FLAGS.rnn_layers):            
            with tf.variable_scope('rhn_' + str(i)):
                lstm_fw = RHNCell(FLAGS.num_hidden, FLAGS.rhn_steps)
                lstm_bw = RHNCell(FLAGS.num_hidden, FLAGS.rhn_steps)
                output, state = tf.nn.bidirectional_dynamic_rnn(lstm_fw, lstm_bw, x , scope='bi_rhn' + str(i), dtype=tf.float32)
                x = tf.concat(output, axis=2)
                print('rhn_'+ str(i) +':  ', x.get_shape())
            
        shape = tf.shape(x) 
        #batch_s, max_timesteps = shape[0], shape[1] 
        outputs = tf.reshape(x, [-1, 2 * FLAGS.num_hidden])

        W = tf.Variable(tf.truncated_normal([2 * FLAGS.num_hidden, num_classes], stddev=0.1), name="w_dense") 
        b = tf.Variable(tf.constant(0., shape=[num_classes]), name="b_dense") 

        # Doing the affine projection 
        self.logits = tf.matmul(outputs, W) + b
            # Reshaping back to the original shape
        shape = tf.shape(x)
        self.logits = tf.reshape(self.logits, [FLAGS.batch_size, -1, num_classes])
            # Time major
        self.logits = tf.transpose(self.logits, (1, 0, 2))

        
        
    def _build_train_op(self):
        self.global_step = tf.Variable(0, trainable=False)

        self.loss = tf.nn.ctc_loss(labels=self.labels,
                                   inputs=self.logits,
                                   sequence_length=self.seq_len)
        self.cost = tf.reduce_mean(self.loss)
        tf.summary.scalar('cost', self.cost)

        self.lrn_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                                   self.global_step,
                                                   FLAGS.decay_steps,
                                                   FLAGS.decay_rate,
                                                   staircase=True)

        # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lrn_rate,
        #                                            momentum=FLAGS.momentum).minimize(self.cost,
        #                                                                              global_step=self.global_step)
        # self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.lrn_rate,
        #                                             momentum=FLAGS.momentum,
        #                                             use_nesterov=True).minimize(self.cost,
        #                                                                         global_step=self.global_step)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.initial_learning_rate,
                                                beta1=FLAGS.beta1,
                                                beta2=FLAGS.beta2).minimize(self.loss,
                                                                            global_step=self.global_step)
        train_ops = [self.optimizer] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)

        # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
        # (it's slower but you'll get better results)
        # decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len,merge_repeated=False)
        self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(self.logits,
                                                                    self.seq_len,
                                                                    merge_repeated=False)
        self.dense_decoded = tf.sparse_tensor_to_dense(self.decoded[0], default_value=-1)

    def _conv2d(self, x, name, filter_size, in_channels, out_channels, strides):
        with tf.variable_scope(name):
            kernel = tf.get_variable(name='DW',
                                     shape=[filter_size[0], filter_size[1], in_channels, out_channels],
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer())

            b = tf.get_variable(name='bais',
                                shape=[out_channels],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer())

            con2d_op = tf.nn.conv2d(x, kernel, [1, strides[0], strides[1], 1], padding='SAME')

        return tf.nn.bias_add(con2d_op, b)

    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]

            beta = tf.get_variable(
                'beta', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable(
                'gamma', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32))

            if self.mode == 'train':
                mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

                moving_mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                moving_variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)

                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_mean, mean, 0.9))
                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_variance, variance, 0.9))
            else:
                mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)

                tf.summary.histogram(mean.op.name, mean)
                tf.summary.histogram(variance.op.name, variance)
            # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
            x_bn = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
            x_bn.set_shape(x.get_shape())

            return x_bn

    def _leaky_relu(self, x, leakiness=0.0):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _max_pool(self, x, ksize, strides):
        return tf.nn.max_pool(x,
                              ksize=[1, ksize[0], ksize[1], 1],
                              strides=[1, strides[0], strides[1], 1],
                              padding='SAME',
                              name='max_pool')

