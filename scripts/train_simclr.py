'''
Created Feb 2022
author: landeros10
Christian Landeros, PhD

Hakho Lee Laboratory
Center for Systems Biology
Massachusetts General Hospital

Massachusetts Institute of Technology

Main SimCLR Training Pipeline
'''
import tensorflow as tf
from tensorflow.linalg import matmul
import logging
import argparse

from ..models.simclr import SimCLRKerasModel
from .util import build_optimizer, build_simclr_loss
from .train_simclr import build_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# TODO
# double check default values
parser = argparse.ArgumentParser()

# Model Definitions
parser.add_argument('--input_s', type=int, default=256, help='Size of the image input')
parser.add_argument('--resnet_depth', type=int, default=50, help='Depth of the ResNet model')
parser.add_argument('--weight_init', type=str, default='imagenet', help='Type of weight initialization')
parser.add_argument('--resnet_pooling', type=str, default='avg', help='Type of pooling in ResNet')
parser.add_argument('--proj_out_dim', type=int, default=128, help='Output dimension after projection')
parser.add_argument('--proj_n_layers', type=int, default=2, help='Number of hidden layers')

parser.add_argument('--bs', type=int, default=256, help='Batch size')
parser.add_argument('--optimizer', type=str, default='adam', help='Type of optimizer')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--warmup_steps', type=int, default=0, help='Number of warmup steps')
parser.add_argument('--clip_norm', type=float, default=0.0, help='Clip norm value')

parser.add_argument('--cpc_temperature', type=float, default=0.1, help='CPC temperature')
parser.add_argument('--l2_coeff', type=float, default=0.0, help='L2 regularization coefficient')

parser.add_argument('--cpc_perturbation', type=float, default=0.1, help='CPC perturbation')
parser.add_argument('--cpc_loss_coeff', type=float, default=1.0, help='CPC loss coefficient')

FLAGS, _ = parser.parse_known_args()


def build_simclr_loss(hp, model, strategy, eps, cpc_loss_coeff, l2_coeff):
    """Builds the SimCLR loss function.

    Args:
        hp: Hyperparameters dictionary.
        model: SimCLR model.
        strategy: A `tf.distribute.Strategy` object.
        eps: Perturbation value.
        tau: Temperature value.
        cpc_loss_coeff: Coefficient for the CPC loss.
        l2_coeff: Coefficient for the L2 regularization loss.

    Returns:
        The SimCLR loss function.
    """
    cce_loss = tf.nn.softmax_cross_entropy_with_logits
    def loss_fn(y_true, y_pred):
        logits1 = y_pred['logits1']
        logits2 = y_pred['logits2']
        labels = y_pred['labels']

        if eps > 0:
            p = (tf.ones_like(labels) * eps) - (2 * labels * eps)
            logits1 = logits1 + p
            logits2 = logits2 + p

        loss1 = cce_loss(labels, logits1) * cpc_loss_coeff
        loss2 = cce_loss(labels, logits2) * cpc_loss_coeff
        cpc_loss = tf.reduce_mean(loss1 + loss2)
        return cpc_loss
    return loss_fn


class ContrastiveAccuracy(tf.keras.metrics.Mean):
    def __init__(self, name='contrastive_accuracy', **kwargs):
        super(ContrastiveAccuracy, self).__init__(**kwargs)

    def update_state(self, logits, sample_weight=None):
        probabilities = tf.nn.softmax(logits)
        bs_replica = tf.shape(logits)[0]
        bs_global = tf.shape(logits)[1]

        replica_context = tf.distribute.get_replica_context()
        replica_id = tf.cast(tf.cast(replica_context.replica_id_in_sync_group,
                                     tf.uint32),
                             tf.int32)
        labels_idx = tf.range(bs_replica, dtype=tf.int32) + replica_id * bs_replica
        masks = tf.one_hot(labels_idx, bs_global, dtype=probabilities.dtype)
        acc_masks = tf.reduce_sum(masks * probabilities) / float(bs_replica)
        super().update_state(acc_masks, sample_weight=sample_weight)


class ContrastiveAccuracy(tf.keras.metrics.Mean):
    def __init__(self, name='contrastive_accuracy', **kwargs):
        super(ContrastiveAccuracy, self).__init__(**kwargs)

    def update_state(self, logits, sample_weight=None):
        probabilities = tf.nn.softmax(logits)
        bs_replica = tf.shape(logits)[0]
        bs_global = tf.shape(logits)[1]

        replica_context = tf.distribute.get_replica_context()
        replica_id = tf.cast(tf.cast(replica_context.replica_id_in_sync_group,
                                     tf.uint32),
                             tf.int32)
        labels_idx = tf.range(bs_replica, dtype=tf.int32) + replica_id * bs_replica
        masks = tf.one_hot(labels_idx, bs_global, dtype=probabilities.dtype)
        acc_masks = tf.reduce_sum(masks * probabilities) / float(bs_replica)
        super().update_state(acc_masks, sample_weight=sample_weight)


class ContrastiveEntropy(tf.keras.metrics.Mean):
    def __init__(self, name='contrastive_entropy', **kwargs):
        super(ContrastiveEntropy, self).__init__(**kwargs)

    def update_state(self, values, sample_weight=None):
        probabilities = tf.nn.softmax(values)
        entropy_con = -tf.reduce_mean(
            tf.reduce_sum(probabilities * tf.math.log(probabilities + 1e-8), -1))
        return super().update_state(entropy_con, sample_weight)


class L2Metric(tf.keras.metrics.Mean):
    def __init__(self, name='l2_loss', **kwargs):
        super(L2Metric, self).__init__(**kwargs)

    def update_state(self, values, sample_weight=None):
        probabilities = tf.nn.softmax(values)
        entropy_con = -tf.reduce_mean(
            tf.reduce_sum(probabilities * tf.math.log(probabilities + 1e-8), -1))
        return super().update_state(entropy_con, sample_weight)


def build_model(hp):
    """
    Hyper-parameters:
    input_s : int, size of the image input (s x s x 3). Default is 224.
    resnet_depth : int, depth of the ResNet model. Default is 50.
    weight_init : str, type of weight initialization. Default is 'imagenet'.
    resnet_pooling : str, type of pooling in resnet. Default is 'avg'.
    proj_out_dim : int, output dimension after projection. Default is 128.
    proj_n_layers : int, number of hidden layers. Default is 2.
    """

    input_s = hp.get('input_s', FLAGS.input_s)  # default value is 256
    resnet_depth = hp.get('resnet_depth', FLAGS.resnet_depth)  # default value is 50
    weight_init = hp.get('weight_init', FLAGS.weight_init)  # default value is 'imagenet'
    resnet_pooling = hp.get('resnet_pooling', FLAGS.resnet_pooling)  # default value is 'avg'
    proj_out_dim = hp.get('proj_out_dim', FLAGS.proj_out_dim)  # default value is 128
    proj_n_layers = hp.get('proj_n_layers', FLAGS.proj_n_layers)  # default value is 2

    l2_coeff = hp.get("l2_coeff", FLAGS.l2_coeff)
    cpc_temperature = hp.get('cpc_temperature', FLAGS.cpc_temperature)

    input_shape = (input_s, input_s, 3)
    model = SimCLRKerasModel(
        input_shape,
        resnet_depth,
        weight_init,
        resnet_pooling,
        proj_out_dim,
        proj_n_layers,
        l2_coeff=l2_coeff,
        cpc_temperature=cpc_temperature,
        name="SimCLR_model",
    )
    return model


def train_model(hp, train_dataset, val_dataset, train_steps, steps_per_epoch):
    # Training Strategy
    strategy = hp.get('strategy', tf.distribute.get_strategy())

    with strategy.scope():
        model = build_model(hp)

        # Optimizer
        bs = hp.get('bs', FLAGS.bs)
        optimizer_type = hp.get("optimizer", FLAGS.optimizer)
        lr = hp.get("learning_rate", FLAGS.learning_rate)
        warmup_steps = hp.get("warmup_steps", FLAGS.warmup_steps)
        if warmup_steps > 0:
            lr = tf.keras.optimizers.schedules.CosineDecay(
                1e-6,
                train_steps,
                alpha=1e-6,
                name='cosine_decay_scheduler',
                warmup_target=lr * float(bs) / 256,
                warmup_steps=warmup_steps
            )
        clip_norm = hp.get("clip_norm", FLAGS.clip_norm)
        clip_norm = float(clip_norm) if clip_norm > 0 else None
        optimizer = build_optimizer(optimizer_type, lr, clip_norm)
        
        # Loss function
        eps = hp.get('cpc_perturbation', FLAGS.cpc_perturbation)
        cpc_loss_coeff = hp.get("cpc_loss_coeff", FLAGS.cpc_loss_coeff)
        l2_coeff = hp.get("l2_coeff", FLAGS.l2_coeff)
        loss_fn = build_simclr_loss(hp, model, strategy, eps, cpc_loss_coeff, l2_coeff)

        # Metrics
        metrics = {
            'logits12': [ContrastiveAccuracy(name='cpc_acc'), ContrastiveEntropy(name='cpc_entropy')],
            }

        model.compile(optimizer=optimizer,
                      loss=loss_fn,
                      metrics=metrics)
        
        # Train Steps
        num_epochs = train_steps // steps_per_epoch

        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=num_epochs,
            steps_per_epoch=train_steps
        )

    return model, history


if __name__ == "__main__":
    # Set up your train and validation datasets
    train_dataset = ...
    val_dataset = ...
    
    # Set up hyperparameters
    hp = {
        "strategy": tf.distribute.get_strategy(),
        "optimizer": "adam",
        "learning_rate": 0.001,
        "use_warmup": False,
        "clip_norm": 0.0
    }
    
    # Set up the number of training and validation steps
    train_steps = ...
    val_steps = ...
    
    # Train the model
    model, history = train_model(hp, train_dataset, val_dataset, train_steps, val_steps)
    
    # Do something with the trained model and history
    ...