'''
Created Feb 2022
author: landeros10
Christian Landeros, PhD

Hakho Lee Laboratory
Center for Systems Biology
Masachusetts General Hospital

Massachusetts Institute of Technology

SimCLRv2 in Tensorflow 2.0 with IFM
Adapted from Chen et al.
http://github.com/google-research/simclr

'''
import tensorflow as tf
from tensorflow.keras.applications import resnet50, resnet # type: ignore
from tensorflow.keras.initializers import Constant, HeNormal
from tensorflow.keras.regularizers import L2
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, PReLU
from tensorflow.linalg import matmul
import logging

from ..scripts.util import gpu_cross_replica_concat

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
LARGE = 1e9

resnet_selector = {
    50: resnet50.ResNet50,
    101: resnet.ResNet101,
    152: resnet.ResNet152,
}

class ProjectionHead(tf.keras.layers.Layer):
    def __init__(self, proj_in_dim, proj_out_dim, proj_n_layers, l2_coeff=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.in_dim = proj_in_dim
        self.out_dim = proj_out_dim
        self.proj_n_layers = proj_n_layers
        self.l2_coeff = l2_coeff
        self.linear_layers = []

        for j in range(self.proj_n_layers):
            if j != self.proj_n_layers - 1:
                # for the middle layers, use bias and relu for the output.
                self.linear_layers.append(
                    tf.keras.Sequential([
                        Dense(
                            self.in_dim,
                            kernel_initializer=HeNormal(), use_bias=True,
                            kernel_regularizer=L2(self.l2_coeff)),
                        BatchNormalization(momentum=0.9, epsilon=1e-5),
                        Activation('gelu'),
                    ])
                )
            else:
                self.linear_layers.append(
                    tf.keras.Sequential([
                        Dense(
                            self.out_dim,
                            kernel_initializer=HeNormal(), use_bias=False,
                            kernel_regularizer=L2(self.l2_coeff)),
                        BatchNormalization(momentum=0.9, epsilon=1e-5),
                    ])
                )

        self.prelu = PReLU()

    def call(self, inputs, training):
        hiddens_list = [tf.identity(inputs, 'proj_head_input')]
        for j in range(self.proj_n_layers):
            hiddens = self.linear_layers[j](hiddens_list[-1], training)
            hiddens_list.append(hiddens)
        proj_head_output = tf.identity(hiddens_list[-1], 'proj_head_output')
        return proj_head_output, hiddens_list[0]


class SimCLRKerasModel(tf.keras.Model):
    def __init__(self, input_shape,
                 resnet_depth, weight_init, resnet_pooling,
                 proj_out_dim, proj_n_layers,
                 l2_coeff=1e-6, cpc_temperature=0.1,
                 **kwargs):
        """
        SimCLRKerasModel is a custom Keras model for SimCLR (Contrastive Learning) implementation.

        Args:
            input_shape (tuple): The shape of the input images.
            resnet_depth (int): The depth of the ResNet model to be used.
            weight_init (str): The weight initialization method for the ResNet model.
            resnet_pooling (str): The pooling method to be used in the ResNet model.
            proj_out_dim (int): The output dimension of the projection head.
            proj_n_layers (int): The number of layers in the projection head.
            **kwargs: Additional keyword arguments to be passed to the parent class.

        """
        super().__init__(**kwargs)
        self.strategy = tf.distribute.get_strategy()
        self.cpc_temperature = cpc_temperature

        self.input_s = input_shape[0]
        resnet_builder = resnet_selector[resnet_depth]
        self.resnet_model = resnet_builder(
            include_top=False,
            weights=weight_init,
            input_tensor=tf.keras.Input(input_shape),
            input_shape=input_shape,
            pooling=resnet_pooling,
        )
        proj_in_dim = 2048

        self.proj_out_dim = proj_out_dim
        self.proj_n_layers = proj_n_layers
        self.projection_model = ProjectionHead(
            proj_in_dim,
            self.proj_out_dim,
            self.proj_n_layers,
            l2_coeff=l2_coeff,
        )

    def call(self, inputs, training=False):
        """Call to SimCLR model. images is an iterable containing the following
        (input image tensor, label tensor or None)

        Args:
            inputs (tf.Tensor): The input tensor containing the images.
            training (bool): Whether the model is in training mode or not.

        Returns:
            tuple: A tuple containing the projection output tensor and the hidden tensor.

        """
        # if tf.rank(inputs) > 4:
        #     # Reshape doubled augmentations
        #     aug1, aug2 = tf.split(inputs, 2, axis=1)
        #     aug1 = tf.squeeze(aug1, axis=1)
        #     aug2 = tf.squeeze(aug2, axis=1)
        #     inputs = tf.concat([aug1, aug2], axis=0)
        #
        hiddens = self.resnet_model(inputs, training)
        proj, hiddens = self.projection_model(hiddens, training)

        # Prepare for self-supervised task
        logits1, logits2, logits12, labels = self.contrastive_to_logits(proj)

        return {
            'proj': proj,
            'hiddens': hiddens,
            'logits1': logits1,
            'logits2': logits2,
            'logits12': logits12,
            'labels': labels
        }
    
    def contrastive_to_logits(self, proj):
        tau = self.cpc_temperature

        bs = tf.shape(x)[0] // 2
        x = tf.math.l2_normalize(x, -1)
        x1, x2 = tf.split(x, 2, 0)

        x1_large = gpu_cross_replica_concat(x1, self.strategy)
        x2_large = gpu_cross_replica_concat(x2, self.strategy)
        enlarged_batch_size = tf.shape(x1_large)[0]
        replica_context = tf.distribute.get_replica_context()
        replica_id = tf.cast(tf.cast(replica_context.replica_id_in_sync_group,
                                     tf.uint32),
                             tf.int32)
        labels_idx = tf.range(bs, dtype=tf.int32) + replica_id * bs
        labels = tf.one_hot(labels_idx, enlarged_batch_size * 2,
                            dtype=x1_large.dtype)
        masks = tf.one_hot(labels_idx, enlarged_batch_size,
                           dtype=x1_large.dtype)

        # Perturbation
        # Subtract (masks * LARGE) so that
        logits11 = matmul(x1, x1_large, transpose_b=True) / tau
        logits11 = logits11 - (masks * LARGE)
        logits12 = matmul(x1, x2_large, transpose_b=True) / tau
        logits1 = tf.concat([logits12, logits11], axis=-1)

        logits22 = matmul(x2, x2_large, transpose_b=True) / tau
        logits22 = logits22 - masks * LARGE
        logits21 = matmul(x2, x1_large, transpose_b=True) / tau
        logits2 = tf.concat([logits21, logits22], axis=-1)

        return logits1, logits2, logits12, labels
