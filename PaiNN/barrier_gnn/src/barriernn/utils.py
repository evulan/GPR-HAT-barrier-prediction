
import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
# from tensorflow.keras.losses import Loss
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend

def get_distance(array1, array2):
    c = array2 - array1
    d = np.sqrt(np.sum(np.square(c)))
    return d

# class MALE(Loss):
#     def __init__(self):
#         super().__init__()
#     def call(self, y_true, y_pred):
#         y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
#         y_true = math_ops.cast(y_true, y_pred.dtype)
#         first_log = math_ops.log(backend.maximum(y_pred, backend.epsilon()) + 1.)
#         second_log = math_ops.log(backend.maximum(y_true, backend.epsilon()) + 1.)
#         ale = math_ops.abs(first_log + math_ops.neg(second_log))
#         return backend.mean(ale, axis=-1)

@tf.keras.utils.register_keras_serializable(package='barriernn', name='MALE')
def MALE(y_true, y_pred):
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    first_log = math_ops.log(backend.maximum(y_pred, backend.epsilon()) + 1.)
    second_log = math_ops.log(backend.maximum(y_true, backend.epsilon()) + 1.)
    ale = math_ops.abs(first_log + math_ops.neg(second_log))
    return backend.mean(ale, axis=-1)

class ExponentialLearningRateScheduler(tf.keras.callbacks.LearningRateScheduler):
    """Callback for exponential change of the learning rate every x epochs."""

    def __init__(self, learning_rate_start=1e-3, learning_rate_stop=0.5e-5, alpha = 0.5, x=10, epo_min=0, epo=500, verbose=0):
        super(ExponentialLearningRateScheduler, self).__init__(schedule=self.schedule_implement, verbose=verbose)
        self.learning_rate_start = learning_rate_start
        self.learning_rate_stop = learning_rate_stop
        self.alpha = alpha
        self.x = x
        self.epo = epo
        self.epo_min = epo_min

    def schedule_implement(self, epoch, lr):
        out = float(max(self.learning_rate_stop, float(self.alpha**((epoch-self.epo_min)//self.x) * self.learning_rate_start)))
        assert tf.summary.scalar('learning rate manual', data=out, step=epoch)
        return float(out)

    def get_config(self):
        config = super(ExponentialLearningRateScheduler, self).get_config()
        config.update({"learning_rate_start": self.learning_rate_start, "learning_rate_stop": self.learning_rate_stop, "alpha": self.alpha, \
            "rate": self.x, "epo": self.epo, "epo_min":self.epo_min})
        return config

class CosineLearningRateScheduler(tf.keras.callbacks.LearningRateScheduler):
    """Callback for cosine change of the learning rate."""

    def __init__(self, learning_rate_start=1e-3, learning_rate_stop=0.5e-5, epo_min=0, epo=500, verbose=0):
        super(CosineLearningRateScheduler, self).__init__(schedule=self.schedule_implement, verbose=verbose)
        self.learning_rate_start = learning_rate_start
        self.learning_rate_stop = learning_rate_stop
        self.epo = epo
        self.epo_min = epo_min

    def schedule_implement(self, epoch, lr=None):
        out = float(self.learning_rate_stop + (1+np.cos(np.pi*(epoch/self.epo)))*(self.learning_rate_start-self.learning_rate_stop)/2)
        return float(out)

    def get_config(self):
        config = super(CosineLearningRateScheduler, self).get_config()
        config.update({"learning_rate_start": self.learning_rate_start, "learning_rate_stop": self.learning_rate_stop, "epo": self.epo, "epo_min":self.epo_min})
        return config


class LinearWarmupCosineLearningRateScheduler(tf.keras.callbacks.LearningRateScheduler):

    def __init__(self, learning_rate_start=1e-3, learning_rate_stop=0.5e-5, epo_min=0, epo=500, verbose=0):
        super(LinearWarmupCosineLearningRateScheduler, self).__init__(schedule=self.schedule_implement, verbose=verbose)
        self.learning_rate_start = learning_rate_start
        self.learning_rate_stop = learning_rate_stop
        self.epo = epo
        self.epo_min = epo_min

    def schedule_implement(self,epoch,lr):
        if epoch < self.epo_min:
            new_rate = self.learning_rate_stop + (self.learning_rate_start - self.learning_rate_stop)/(self.epo_min-epoch)
            out = min(self.learning_rate_start, new_rate)
        else:    
            out = self.learning_rate_stop + (1+np.cos(np.pi*(epoch/self.epo)))*(self.learning_rate_start-self.learning_rate_stop)/2
        return float(out)
    def get_config(self):
        config = super(LinearWarmupCosineLearningRateScheduler, self).get_config()
        config.update({"learning_rate_start": self.learning_rate_start, "learning_rate_stop": self.learning_rate_stop, "epo": self.epo, "epo_min":self.epo_min})
        return config

class LinearWarmupExpLearningRateScheduler(tf.keras.callbacks.LearningRateScheduler):
    def __init__(self, learning_rate_start=1e-3, learning_rate_stop=0.5e-5, alpha = 0.5, x=10, epo_min=0, epo=500, verbose=0):
        super(LinearWarmupExpLearningRateScheduler, self).__init__(schedule=self.schedule_implement, verbose=verbose)
        self.learning_rate_start = learning_rate_start
        self.learning_rate_stop = learning_rate_stop
        self.alpha = alpha
        self.x = x
        self.epo = epo
        self.epo_min = epo_min

    def schedule_implement(self,epoch,lr):
        if epoch < self.epo_min:
            new_rate = self.learning_rate_stop + (self.learning_rate_start - self.learning_rate_stop)/(self.epo_min-epoch)
            out = float(min(self.learning_rate_start, new_rate))
        else:    
            out = float(max(self.learning_rate_stop, float(self.alpha**((epoch-self.epo_min)//self.x) * self.learning_rate_start)))
        return float(out)
    def get_config(self):
        config = super(LinearWarmupExpLearningRateScheduler, self).get_config()
        config.update({"learning_rate_start": self.learning_rate_start, "learning_rate_stop": self.learning_rate_stop, "alpha": self.alpha, \
            "rate": self.x, "epo": self.epo, "epo_min":self.epo_min})
        return config


def generate_standard_graph_input(input_node_shape,
                                  input_edge_shape,
                                  input_state_shape,
                                  input_node_vocab=95,
                                  input_edge_vocab=5,
                                  input_state_vocab=100,
                                  input_node_embedd=64,
                                  input_edge_embedd=64,
                                  input_state_embedd=64,
                                  input_tensor_type='ragged'):
    """Generate input for a standard graph tensor format.
    This includes nodes, edge, edge_indices and optional a graph state.
    If input shape is (None,) a embedding layer is used to make the feature dimension.
    Args:
        input_node_shape (list): Shape of node input without batch dimension. Either (None,F) or (None,)
        input_edge_shape (list): Shape of edge input without batch dimension. Either (None,F) or (None,)
        input_state_shape: Shape of state input without batch dimension. Either (F,) or (,)
        input_node_vocab (int): Vocabulary size of optional embedding layer.
        input_edge_vocab (int): Vocabulary size of optional embedding layer.
        input_state_vocab (int) Vocabulary size of optional embedding layer.
        input_node_embedd (int): Embedding dimension for optional embedding layer.
        input_edge_embedd (int): Embedding dimension for optional embedding layer.
        input_state_embedd (int): Embedding dimension for optional embedding layer.
        input_tensor_type (str): Type of input tensor. Only "ragged" is supported at the moment.
    Returns:
        list: [node_input, node_embedding, edge_input, edge_embedding, edge_index_input, state_input, state_embedding]
    """
    node_input = ks.layers.Input(shape=input_node_shape, name='node_input', dtype="float32", ragged=True)
    edge_input = ks.layers.Input(shape=input_edge_shape, name='edge_input', dtype="float32", ragged=True)
    edge_index_input = ks.layers.Input(shape=(None, 2), name='edge_index_input', dtype="int64", ragged=True)

    if len(input_node_shape) == 1:
        n = ks.layers.Embedding(input_node_vocab, input_node_embedd, name='node_embedding')(node_input)
    else:
        n = node_input

    if len(input_edge_shape) == 1:
        ed = ks.layers.Embedding(input_edge_vocab, input_edge_embedd, name='edge_embedding')(edge_input)
    else:
        ed = edge_input

    if input_state_shape is not None:
        env_input = ks.Input(shape=input_state_shape, dtype='float32', name='state_input')
        if len(input_state_shape) == 0:
            uenv = ks.layers.Embedding(input_state_vocab, input_state_embedd, name='state_embedding')(env_input)
        else:
            uenv = env_input

    if input_state_shape is not None:
        return node_input, n, edge_input, ed, edge_index_input, env_input, uenv,
    else:
        return node_input, n, edge_input, ed, edge_index_input, None, None


def update_model_args(default_args=None, user_args=None):
    """
    Make arg dict with updated default values.
    Args:
        default_args (dict): Dictionary of default values.
        user_args (dict): Dictionary of args from.
    Returns:
        dict: Make new dict and update with first default and then user args.
    """
    out = {}
    if default_args is None:
        default_args = {}
    if user_args is None:
        user_args = {}
    out.update(default_args)
    out.update(user_args)
    return out


def generate_mol_graph_input(input_node_shape,
                             input_xyz_shape,
                             input_bond_index_shape=None,
                             input_angle_index_shape=None,
                             input_dihedral_index_shape=None,
                             input_node_vocab=95,
                             input_node_embedd=64,
                             input_tensor_type='ragged'):
    """Generate input for a standard graph tensor format.
    This includes nodes, edge, edge_indices and optional a graph state.
    If input shape is (None,) a embedding layer is used to make the feature dimension.
    Args:
        input_node_shape (list): Shape of node input without batch dimension. Either (None,F) or (None,)
        input_xyz_shape (list): Shape of xyz input without batch dimension (None,3).
        input_node_vocab (int): Vocabulary size of optional embedding layer.
        input_node_embedd (int): Embedding dimension for optional embedding layer.
        input_tensor_type (str): Type of input tensor. Only "ragged" is supported at the moment.
    Returns:
        list: [node_input, node_embedding, edge_input, edge_embedding, edge_index_input, state_input, state_embedding]
    """
    node_input = ks.layers.Input(shape=input_node_shape, name='node_input', dtype="float32", ragged=True)
    xyz_input = ks.layers.Input(shape=input_xyz_shape, name='xyz_input', dtype="float32", ragged=True)

    if input_bond_index_shape is not None:
        bond_index_input = ks.layers.Input(shape=input_bond_index_shape, name='bond_index_input', dtype="int64",
                                           ragged=True)
    else:
        bond_index_input = None

    if input_angle_index_shape is not None:
        angle_index_input = ks.layers.Input(shape=input_angle_index_shape, name='angle_index_input', dtype="int64",
                                            ragged=True)
    else:
        angle_index_input = None

    if input_dihedral_index_shape is not None:
        dihedral_index_input = ks.layers.Input(shape=input_dihedral_index_shape, name='dihedral_index_input', dtype="int64",
                                               ragged=True)
    else:
        dihedral_index_input = None

    if len(input_node_shape) == 1:
        n = ks.layers.Embedding(input_node_vocab, input_node_embedd, name='node_embedding')(node_input)
    else:
        n = node_input

    return node_input, n, xyz_input, bond_index_input, angle_index_input, dihedral_index_input

