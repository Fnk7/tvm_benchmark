import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import graph_util

from tvm import relay
import tvm.relay.testing.tf as tf_testing

try:
    tf = tf.compat.v1
except:
    tf = tf



class LSTMCell(object):
    W = []
    U = []
    b = []
    def __init__(self, hidden_size, scope):
        with tf.variable_scope(scope):
            self.W = []
            self.U = []
            self.b = []
            self.num_unit = hidden_size
            for i in range(4):
                W = tf.get_variable(
                    "W%d" % (i), [self.num_unit, self.num_unit], dtype=tf.float32)
                U = tf.get_variable(
                    "U%d" % (i), [self.num_unit, self.num_unit], dtype=tf.float32)
                b = tf.get_variable("bias%d" % (i), [self.num_unit], dtype=tf.float32,
                                    initializer=init_ops.constant_initializer(0, dtype=tf.float32))
                self.W.append(W)
                self.U.append(U)
                self.b.append(b)

    def call(self, inputs, state):
        c, h = state
        res = []
        for i in range(4):
            res.append(math_ops.matmul(
                inputs, self.W[i]) + math_ops.matmul(h, self.U[i]) + self.b[i])
        i, j, f, o = (res[0], res[1], res[2], res[3])
        new_c = (c * math_ops.sigmoid(f + 1.0) +
                 math_ops.sigmoid(i) * math_ops.tanh(j))
        new_h = math_ops.tanh(new_c) * math_ops.sigmoid(o)
        new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)
        return new_h, new_state


class LSTMModel(object):
    stacked_cells = []

    def __init__(self, num_layer, hidden_size):
        self.stacked_cells = []
        self.num_layer = num_layer
        self.num_unit = hidden_size
        for layer in range(self.num_layer):
            self.stacked_cells.append(
                LSTMCell(self.num_unit, "LSTMLayer%d" % (layer)))

    def run(self, inputs, batch_size, num_step):
        self.batch_size = batch_size
        self.num_step = num_step

        cell = tf.nn.rnn_cell.BasicLSTMCell(
            self.num_unit, forget_bias=1.0, state_is_tuple=True)
        self._initial_state = cell.zero_state(batch_size, tf.float32)
        self.state = [self._initial_state for layer in range(self.num_layer)]

        for step in range(self.num_step):
            cur_input = inputs[step, :, :]
            for layer in range(self.num_layer):
                cell_output, self.state[layer] = self.stacked_cells[layer].call(
                    cur_input, self.state[layer])
                cur_input = cell_output

        self.output = cell_output
        return self.output, self.state[-1]


def get_workload(**kwargs):
    num_layer = 3 if "num_layer" not in kwargs else kwargs["num_layer"]
    num_step = 10 if "num_step" not in kwargs else kwargs["num_step"]
    batch_size = 1 if "batch_size" not in kwargs else kwargs["batch_size"]
    hidden_size = 128 if "hidden_size" not in kwargs else kwargs["hidden_size"]
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        graph_options=tf.GraphOptions(infer_shapes=True),
        inter_op_parallelism_threads=False
    )
    with tf.Graph().as_default(), tf.Session(config=session_conf) as session:
        model = LSTMModel(num_layer, hidden_size)
        inputs = tf.placeholder(tf.float32, [num_step, batch_size, hidden_size], 'data')
        shape_dict = {'data': [num_step, batch_size, hidden_size]}
        lstm_output = model.run(inputs, batch_size, num_step)[0]
        session.run(tf.global_variables_initializer())
        output_node = [lstm_output.name.split(':')[0]]
        constant_graph = graph_util.convert_variables_to_constants(
                    session, session.graph_def, [lstm_output.name.split(':')[0]])
        graph_def = tf_testing.ProcessGraphDefParam(constant_graph)
        graph_def = tf_testing.AddShapesToGraphDef(session, output_node[0])
    mod, params = relay.frontend.from_tensorflow(graph_def, shape=shape_dict, outputs=output_node)
    return mod, params, shape_dict['data'], lstm_output.shape.as_list()


if __name__ == "__main__":
    mod, params, _, _ = get_workload()
    from compile import compile
    compile(mod, params, "lstm", True, False, False)

