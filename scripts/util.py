import tensorflow as tf
from tensorflow.data import TFRecordDataset
from tensorflow.keras import optimizers
import numpy as np
from PIL import Image
from openslide import OpenSlide
import albumentations as A

import logging
from os.path import join
from os import cpu_count, environ
from glob import glob
import re

from scripts.randstainna import RandStainNA

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# LARS Optimizer
EETA_DEFAULT = 0.001

# Stain Augmentation
yaml_file = '/home/ubuntu/notebooks/cpc_hist/src/CRC_LAB_randomTrue_n0.yaml'
stain_augmentor = RandStainNA(yaml_file, std_hyper=-0.0)
stain_normalizer = RandStainNA(yaml_file, std_hyper=-1.0)


def gpu_cross_replica_concat(tensor, strategy=None):
  """Reduce a concatenation of the `tensor` across TPU cores.

  Args:
    tensor: tensor to concatenate.
    strategy: A `tf.distribute.Strategy`. If not set, CPU execution is assumed.

  Returns:
    Tensor of the same rank as `tensor` with first dimension `num_replicas`
    times larger.
  """
  if strategy is None or strategy.num_replicas_in_sync <= 1:
    return tensor
  gathered = tf.distribute.get_replica_context().all_gather(tensor, 0)
  return gathered


def count_ds_size(ds, bs=1e3):
    count = 0
    for b in ds.batch(int(bs)):
        count += len(b)
    return count


def get_ds_length(ds_files, max_ex):
    N = len(ds_files)

    if max_ex is None:
        ds = TFRecordDataset(ds_files[0])
        max_ex = sum((1 for _ in ds))
    if N == 1:
        return max_ex

    ds = TFRecordDataset(ds_files[-1], num_parallel_reads=cpu_count())
    last_ex = count_ds_size(ds)
    del ds
    return ((N-1) * max_ex) + last_ex


def build_dataset(dataset_dir, patch_size, n_parallel=-1, shuffle=False, max_ex=1e5, return_len=False, verbose=False):
    """ Converts dataset into a distributed dataset according to
    distribute_datasets_from_function """
    logging.info("Using Dataset located at %s", dataset_dir)
    shards = glob(join(dataset_dir, f"dataset_{patch_size}_*.tfrecords"))

    if n_parallel == 0:
        n_parallel = None
    elif n_parallel == -1:
        n_parallel = tf.data.AUTOTUNE
    elif n_parallel == -2:
        n_parallel = cpu_count()

    # ds = TFRecordDataset(shards, num_parallel_reads=n_parallel)
    files_ds = tf.data.Dataset.from_tensor_slices(shards)
    if shuffle:
        files_ds = files_ds.shuffle(len(shards))
    ds = files_ds.interleave(
        lambda x: tf.data.TFRecordDataset(x).repeat(),
        # lambda x: build_dataset_shard(x, patch_size, n_parallel),
        num_parallel_calls=n_parallel,
        deterministic=(not shuffle))
    if verbose:
      logging.info("Data File Path : {}".format(dataset_dir))
      logging.info("Patch Size: {}".format(patch_size))
      logging.info("Total Shards: {}".format(len(shards)))

    total_examples = get_ds_length(shards, max_ex)
    if verbose:
      logging.info("Total Samples: {:.2f}M\n\n".format(total_examples / 1e6))
    if return_len:
        return ds, int(total_examples)
    return ds


def build_optimizer(name, lr, clip_norm=1.0, momentum=0.9, weight_decay=1e-6, decay_steps=None):
    if decay_steps is not None:
        lr = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr,
            decay_steps=decay_steps,
            alpha=1e-3,
            name='cosine_decay_scheduler'
        )

    if name == "momentum":
        print("Using SGD\n")
        optimizer = optimizers.SGD(lr, momentum, nesterov=True, clipnorm=clip_norm)
    elif name == "adam":
        print("Using Adam \n")
        optimizer = optimizers.Adam(lr, clipnorm=clip_norm)
    elif name == "adamw":
        print("Using Adam \n")
        optimizer = optimizers.AdamW(lr, clipnorm=clip_norm)
    elif name == "nadam":
        print("Using Nadam \n")
        optimizer = optimizers.Nadam(lr, clipnorm=clip_norm)
    elif name == "lion":
        print("Using Lion \n")
        optimizer = optimizers.Lion(lr, clipnorm=clip_norm)
    elif name == "lars":
        print("Using LARS \n")
        optimizer = LARSOptimizer(
                lr,
                momentum=momentum,
                weight_decay=weight_decay,
                exclude_from_weight_decay=[
                    'batch_normalization', 'bias', 'head_supervised'
                ],
                clipnorm=clip_norm)
    return optimizer


class LARSOptimizer(tf.keras.optimizers.legacy.Optimizer):
  """Layer-wise Adaptive Rate Scaling for large batch training.

  Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
  I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)
  """

  def __init__(self,
               learning_rate,
               momentum=0.9,
               use_nesterov=False,
               weight_decay=0.0,
               exclude_from_weight_decay=None,
               exclude_from_layer_adaptation=None,
               classic_momentum=True,
               eeta=EETA_DEFAULT,
               name="LARSOptimizer",
               clipnorm=None):
    """Constructs a LARSOptimizer.

    Args:
      learning_rate: A `float` for learning rate.
      momentum: A `float` for momentum.
      use_nesterov: A 'Boolean' for whether to use nesterov momentum.
      weight_decay: A `float` for weight decay.
      exclude_from_weight_decay: A list of `string` for variable screening, if
          any of the string appears in a variable's name, the variable will be
          excluded for computing weight decay. For example, one could specify
          the list like ['batch_normalization', 'bias'] to exclude BN and bias
          from weight decay.
      exclude_from_layer_adaptation: Similar to exclude_from_weight_decay, but
          for layer adaptation. If it is None, it will be defaulted the same as
          exclude_from_weight_decay.
      classic_momentum: A `boolean` for whether to use classic (or popular)
          momentum. The learning rate is applied during momeuntum update in
          classic momentum, but after momentum for popular momentum.
      eeta: A `float` for scaling of learning rate when computing trust ratio.
      name: The name for the scope.
    """
    super(LARSOptimizer, self).__init__(name, clipnorm=clipnorm)

    self._set_hyper("learning_rate", learning_rate)
    self.momentum = momentum
    self.weight_decay = weight_decay
    self.use_nesterov = use_nesterov
    self.classic_momentum = classic_momentum
    self.eeta = eeta
    self.exclude_from_weight_decay = exclude_from_weight_decay
    # exclude_from_layer_adaptation is set to exclude_from_weight_decay if the
    # arg is None.
    if exclude_from_layer_adaptation:
      self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
    else:
      self.exclude_from_layer_adaptation = exclude_from_weight_decay

  def _create_slots(self, var_list):
    for v in var_list:
      self.add_slot(v, "Momentum")

  def _resource_apply_dense(self, grad, param, apply_state=None):
    if grad is None or param is None:
      return tf.no_op()

    var_device, var_dtype = param.device, param.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                    self._fallback_apply_state(var_device, var_dtype))
    learning_rate = coefficients["lr_t"]

    param_name = param.name

    v = self.get_slot(param, "Momentum")

    if self._use_weight_decay(param_name):
      grad += self.weight_decay * param

    if self.classic_momentum:
      trust_ratio = 1.0
      if self._do_layer_adaptation(param_name):
        w_norm = tf.norm(param, ord=2)
        g_norm = tf.norm(grad, ord=2)
        trust_ratio = tf.where(
            tf.greater(w_norm, 0),
            tf.where(tf.greater(g_norm, 0), (self.eeta * w_norm / g_norm), 1.0),
            1.0)
      scaled_lr = learning_rate * trust_ratio

      next_v = tf.multiply(self.momentum, v) + scaled_lr * grad
      if self.use_nesterov:
        update = tf.multiply(self.momentum, next_v) + scaled_lr * grad
      else:
        update = next_v
      next_param = param - update
    else:
      next_v = tf.multiply(self.momentum, v) + grad
      if self.use_nesterov:
        update = tf.multiply(self.momentum, next_v) + grad
      else:
        update = next_v

      trust_ratio = 1.0
      if self._do_layer_adaptation(param_name):
        w_norm = tf.norm(param, ord=2)
        v_norm = tf.norm(update, ord=2)
        trust_ratio = tf.where(
            tf.greater(w_norm, 0),
            tf.where(tf.greater(v_norm, 0), (self.eeta * w_norm / v_norm), 1.0),
            1.0)
      scaled_lr = trust_ratio * learning_rate
      next_param = param - scaled_lr * update

    return tf.group(*[
        param.assign(next_param, use_locking=False),
        v.assign(next_v, use_locking=False)
    ])

  def _use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        # TODO(srbs): Try to avoid name based filtering.
        if re.search(r, param_name) is not None:
          return False
    return True

  def _do_layer_adaptation(self, param_name):
    """Whether to do layer-wise learning rate adaptation for `param_name`."""
    if self.exclude_from_layer_adaptation:
      for r in self.exclude_from_layer_adaptation:
        # TODO(srbs): Try to avoid name based filtering.
        if re.search(r, param_name) is not None:
          return False
    return True

  def get_config(self):
    config = super(LARSOptimizer, self).get_config()
    config.update({
        "learning_rate": self._serialize_hyperparameter("learning_rate"),
        "momentum": self.momentum,
        "classic_momentum": self.classic_momentum,
        "weight_decay": self.weight_decay,
        "eeta": self.eeta,
        "use_nesterov": self.use_nesterov,
    })
    return config


def _parse_function(proto, batched=False, base_path=''):
    keys_to_features = {
                        'file': tf.io.FixedLenFeature([], tf.string),
                        'mag': tf.io.FixedLenFeature([], tf.float32),
                        'input_s': tf.io.FixedLenFeature([], tf.int64),
                        'x': tf.io.FixedLenFeature([], tf.int64),
                        'y': tf.io.FixedLenFeature([], tf.int64),
                        }
    parser = tf.io.parse_example if batched else tf.io.parse_single_example
    parsed_features = parser(proto, keys_to_features)

    if base_path:
        parsed_features['file'] = tf.strings.regex_replace(parsed_features['file'], r'^\./', base_path)
    
    return parsed_features


def _sample_from_tfrecord(pf, out_s, batched=False, return_dict=False, normalize=False):
    """ SVS coords are saved such that x: col, y: row."""
    files = pf["file"]
    xs = pf["x"]
    ys = pf["y"]
    in_ss = pf["input_s"]
    extractor = read_slide_batch if batched else read_slide_
    out_shape = [None, out_s, out_s, 3] if batched else [out_s, out_s, 3]

    tf_arr = tf.py_function(extractor,
                            [files, xs, ys, in_ss, out_s, normalize],
                            Tout=tf.uint8)
    tf_arr.set_shape(out_shape)
    if return_dict:
        pf["images"] = tf_arr
        return pf
    return tf_arr


def read_slide_(file, x, y, in_s, out_s, normalize=False):
    # slideObj = slideObjects[file.numpy().decode("utf-8")]
    if not isinstance(file, str):
        file = file.numpy().decode("utf-8")
    slideObj = OpenSlide(file)
    x, y = int(x), int(y)
    in_s = int(in_s)
    out_s = int(out_s)
    image = slideObj.read_region((x, y), 0, (in_s, in_s)).convert('RGB')
    # image = cv2.resize(image, dsize=(out_s, out_s))
    image = image.resize((out_s, out_s))
    image = np.array(image).astype(np.uint8)
    if normalize:
        image = stain_normalizer(image)
    return image


def read_slide_batch(files, xs, ys, in_ss, out_s, normalize=False):
    bs = xs.shape[0]
    batch_images = np.zeros((2 * bs, out_s, out_s, 3), dtype=np.uint8)

    for i, (file, x, y, in_s) in enumerate(zip(files, xs, ys, in_ss)):
        slideObj = OpenSlide(file.numpy().decode("utf-8"))
        x, y = int(x), int(y)
        image = slideObj.read_region((x, y), 0, (in_s, in_s)).convert('RGB')
        image = np.array(image.resize((out_s, out_s), Image.BILINEAR))
        batch_images[i] = image
        batch_images[i + bs] = image
    return batch_images


def _augment(x, tile_size, bs, num_gpus, aug_params, train=True):
    """ 
    Performs stochastic augmentation steps on input data.

    Parameters:
    - x: Input tensor to augment.
    - tile_size: Size of the tiles to crop to.
    - bs: Batch size for processing.
    - num_gpus: Number of GPUs available, affects batching.
    - aug_params: Dictionary of augmentation parameters.
    - train: If True, apply training augmentations.
    """
    out_size = (bs * 2, tile_size, tile_size, 3)
    
    # Convert aug_params dictionary values to a list of tensors
    aug_params_list = [
        tf.constant(aug_params.get('h_flip_p', 0.5), dtype=tf.float32),
        tf.constant(aug_params.get('v_flip_p', 0.5), dtype=tf.float32),
        tf.constant(aug_params.get('rotate_p', 1.0), dtype=tf.float32),
        tf.constant(aug_params.get('crop_frac', 0.9), dtype=tf.float32),
        tf.constant(aug_params.get('elastic_alpha', 50), dtype=tf.float32),
        tf.constant(aug_params.get('elastic_sigma', 50), dtype=tf.float32),
        tf.constant(aug_params.get('elastic_alpha_affine', 15), dtype=tf.float32),
        tf.constant(aug_params.get('elastic_p', 0.80), dtype=tf.float32),
        tf.constant(aug_params.get('blur_radius', 5), dtype=tf.float32),
        tf.constant(aug_params.get('blur_p', 0.25), dtype=tf.float32),
        tf.constant(aug_params.get('rotate_limit', 10), dtype=tf.float32)
    ]    
    # Use *aug_params_list to unpack the list of tensors as individual arguments
    x = tf.numpy_function(
        func=augmentor_batch,
        inp=[x, tile_size, bs, num_gpus, *aug_params_list],
        Tout=tf.uint8
    )
    
    x.set_shape(out_size)
    x = tf.cast(x, tf.float32) / 255.0
    return x

def augmentor_batch(batch_images, tile_size, bs, num_gpus, 
                    h_flip_p=0.5, v_flip_p=0.5, rotate_p=1.0, crop_frac=0.9, 
                    elastic_alpha=50, elastic_sigma=50, elastic_alpha_affine=15, elastic_p=0.80, 
                    blur_radius=5, blur_p=0.25, rotate_limit=10):
    """
    Augments a batch of images with various transformations using provided parameters.

    Parameters:
    - batch_images: The batch of images to augment.
    - tile_size: Target size for each cropped image tile.
    - bs: Batch size for processing.
    - num_gpus: Number of GPUs for adjusting batch division.
    - h_flip_p: Probability of applying horizontal flip (default: 0.5).
    - v_flip_p: Probability of applying vertical flip (default: 0.5).
    - rotate_p: Probability of applying random rotation of 90 degrees (default: 1.0).
    - crop_frac: Fraction of image to randomly crop and resize (default: 0.9).
    - elastic_alpha: Alpha value for elastic transformation (default: 50).
    - elastic_sigma: Sigma value for elastic transformation (default: 50).
    - elastic_alpha_affine: Alpha affine for elastic transformation (default: 15).
    - elastic_p: Probability of applying elastic transformation (default: 0.80).
    - blur_radius: Radius for the blur effect (default: 5).
    - blur_p: Probability of applying blur (default: 0.25).
    - rotate_limit: Maximum degrees for random rotation (default: 10).
    """
    avg_pixel = batch_images.mean(axis=(0, 1, 2)).astype(int).tolist()

    # Define the augmentation pipeline using Albumentations library
    transform = A.Compose([
        A.HorizontalFlip(h_flip_p),
        A.VerticalFlip(v_flip_p),
        A.RandomRotate90(rotate_p),
        A.RandomResizedCrop(tile_size, tile_size, scale=(crop_frac, 1.0), ratio=(1.0, 1.0), p=1.0),
        A.Lambda(image=stain_augmentor_wrapper),
        A.ElasticTransform(alpha=int(elastic_alpha), sigma=int(elastic_sigma), alpha_affine=int(elastic_alpha_affine), p=elastic_p),
        A.Defocus(radius=(1, int(blur_radius)), alias_blur=0.0, p=blur_p),
        A.ShiftScaleRotate(scale_limit=0.25, rotate_limit=(-int(rotate_limit), int(rotate_limit)), border_mode=0, value=avg_pixel, p=1.0),
    ])
    
    # Apply augmentation transformations
    small_bs = bs // num_gpus
    augmented_images = np.zeros((bs * 2, tile_size, tile_size, 3), dtype=batch_images.dtype)
    for i in range(bs):
        start_idx = (i // small_bs) * small_bs * 2
        aug1_idx = start_idx + (i % small_bs)
        aug2_idx = start_idx + (i % small_bs) + small_bs
        # print(f"bs: {bs}\tidx1:{aug1_idx}\tidx2:{aug2_idx}")

        image = batch_images[i].astype(np.uint8)
        augmented = transform(image=image)['image']
        assert augmented.shape[1] == tile_size
        augmented_images[aug1_idx] = augmented

        augmented = transform(image=image)['image']
        assert augmented.shape[1] == tile_size
        augmented_images[aug2_idx] = augmented
    return augmented_images


def stain_augmentor_wrapper(image, **kwargs):
    if np.std(image) < 10:
        return image
    else:
        return stain_augmentor(image)

