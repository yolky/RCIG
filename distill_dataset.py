import sys

# sys.path.append("..")

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import fire
import ml_collections
from functools import partial

# from jax.config import config
# config.update("jax_enable_x64", True)

import jax
from absl import logging
import absl
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')
from flax.training import train_state, checkpoints


from dataloader import get_dataset, configure_dataloader

# from lib.dataset.dataloader import get_dataset, configure_dataloader
# from lib.models.utils import create_model
# from lib.datadistillation.utils import save_dnfr_image, save_proto_np
# from lib.datadistillation.frepo import proto_train_and_evaluate, init_proto, ProtoHolder
# from lib.training.utils import create_train_state
# from lib.dataset.augmax import get_aug_by_name

from clu import metric_writers

from collections import namedtuple


from models import ResNet18, KIP_ConvNet, linear_net, Conv
from augmax import get_aug_by_name

import numpy as np
import jax.numpy as jnp
import algorithms
import optax
import time
import pickle
import contextlib
import warnings

import json

from jax.config import config as jax_config


def get_config():
    # Note that max_lr_factor and l2_regularization is found through grid search.
    config = ml_collections.ConfigDict()
    config.random_seed = 0
    config.train_log = 'train_log'
    config.train_img = 'train_img'
    config.mixed_precision = False
    config.resume = True

    config.img_size = None
    config.img_channels = None
    config.num_prototypes = None
    config.train_size = None

    config.dataset = ml_collections.ConfigDict()
    config.kernel = ml_collections.ConfigDict()
    config.online = ml_collections.ConfigDict()

    # Dataset
    config.dataset.name = 'cifar100'  # ['cifar10', 'cifar100', 'mnist', 'fashion_mnist', 'tiny_imagenet']
    config.dataset.data_path = 'data/tensorflow_datasets'
    config.dataset.zca_path = 'data/zca'
    config.dataset.zca_reg = 0.1

    # online
    config.online.img_size = None
    config.online.img_channels = None
    config.online.mixed_precision = config.mixed_precision
    config.online.optimizer = 'adam'
    config.online.learning_rate = 0.0003
    config.online.arch = 'dnfrnet'
    config.online.output = 'feat_fc'
    config.online.width = 128
    config.online.normalization = 'identity'

    # Kernel
    config.kernel.img_size = None
    config.kernel.img_channels = None
    config.kernel.num_prototypes = None
    config.kernel.train_size = None
    config.kernel.mixed_precision = config.mixed_precision
    config.kernel.resume = config.resume
    config.kernel.optimizer = 'lamb'
    config.kernel.learning_rate = 0.0003
    config.kernel.batch_size = 1024
    config.kernel.eval_batch_size = 1000

    return config


def main(dataset_name = 'cifar10', data_path=None, zca_path=None, train_log=None, train_img=None, width=128, random_seed=0, message = 'Put your message here!', output_dir = None, n_images = 10, config_path = None, log_dir = None, max_steps = 10000, use_x64 = False, skip_tune = False, naive_loss = False, init_random_noise = False):
    # --------------------------------------
    # Setup
    # --------------------------------------

    if use_x64:
        jax_config.update("jax_enable_x64", True)

    logging.use_absl_handler()

    if log_dir is None and output_dir is not None:
        log_dir = output_dir
    elif log_dir is None:
        log_dir = './logs/'
    
    if not os.path.exists('./{}'.format(log_dir)):
        os.makedirs('./{}'.format(log_dir))

    logging.get_absl_handler().use_absl_log_file('{}, {}'.format(int(time.time()), message), './{}/'.format(log_dir))
    absl.flags.FLAGS.mark_as_parsed() 
    logging.set_verbosity('info')
    
    logging.info('\n\n\n{}\n\n\n'.format(message))
    
    config = get_config()
    config.random_seed = random_seed
    config.train_log = train_log if train_log else 'train_log'
    config.train_img = train_img if train_img else 'train_img'
    # --------------------------------------
    # Dataset
    # --------------------------------------

    

    config.dataset.data_path = data_path if data_path else 'data/tensorflow_datasets'
    config.dataset.zca_path = zca_path if zca_path else 'data/zca'
    config.dataset.name = dataset_name

    (ds_train, ds_test), preprocess_op, rev_preprocess_op, proto_scale = get_dataset(config.dataset)

    coreset_images, coreset_labels = algorithms.init_proto(ds_train, n_images, config.dataset.num_classes, seed = random_seed, random_noise = init_random_noise)

    num_prototypes = n_images * config.dataset.num_classes
    print()
    print(num_prototypes)
    print()
    config.kernel.num_prototypes = num_prototypes
    
    y_transform = lambda y: tf.one_hot(y, config.dataset.num_classes, on_value=1 - 1 / config.dataset.num_classes,
                                           off_value=-1 / config.dataset.num_classes)

    ds_train = configure_dataloader(ds_train, batch_size=config.kernel.batch_size, y_transform=y_transform,
                                        train=True, shuffle=True)
    ds_test = configure_dataloader(ds_test, batch_size=config.kernel.eval_batch_size, y_transform=y_transform,
                                   train=False, shuffle=False)

    
    num_classes = config.dataset.num_classes


    if config.dataset.img_shape[0] in [28, 32]:
        depth = 3
    elif config.dataset.img_shape[0] == 64:
        depth = 4
    elif config.dataset.img_shape[0] == 128:
        depth = 5
    else:
        raise Exception('Invalid resolution for the dataset')
    
    key = jax.random.PRNGKey(random_seed)
    
    alg_config = ml_collections.ConfigDict()


    if config_path is not None:
        print(f'loading config from {config_path}')
        logging.info(f'loading config from {config_path}')
        loaded_dict = json.loads(open('./{}'.format(config_path), 'rb').read())
        loaded_dict['direct_batch_sizes'] = tuple(loaded_dict['direct_batch_sizes'])
        alg_config = ml_collections.config_dict.ConfigDict(loaded_dict)

    alg_config.l2 = alg_config.l2_rate * config.kernel.num_prototypes

    alg_config.use_x64 = use_x64
    alg_config.naive_loss = naive_loss

    alg_config.output_dir = output_dir
    alg_config.max_steps = max_steps
    alg_config.model_depth = depth

    print(alg_config)

    logging.info('using config from ./{}'.format(config_path))
    logging.info(alg_config)

    if output_dir is not None:
        if not os.path.exists('./{}'.format(output_dir)):
            os.makedirs('./{}'.format(output_dir))

        with open('./{}/config.txt'.format(output_dir), 'a') as config_file:
            config_file.write(repr(alg_config))


    model_for_train = Conv(use_softplus = (alg_config.softplus_temp != 0), beta = alg_config.softplus_temp, num_classes = num_classes, width = width, depth = depth, normalization = 'batch' if alg_config.has_bn else 'identity')

    

    

    #Tuning inner and hessian inverse learning rate

    print("Tuning learning rates -- this may take a few minutes")
    logging.info("Tuning learning rates -- this may take a few minutes")

    inner_learning_rate = 0.00001 #initialize them to be small, then gradually increase until unstable
    hvp_learning_rate = 0.00005

    start_time = time.time()

    if not skip_tune:
        with contextlib.redirect_stdout(None):
        # if True:
            inner_result = 1
            while inner_result == 1:
                inner_result, _ = algorithms.run_rcig(coreset_images, coreset_labels, model_for_train.init, model_for_train.apply, ds_train, alg_config, key, inner_learning_rate, hvp_learning_rate, lr_tune = True)
                inner_learning_rate *= 1.2

            inner_learning_rate *= 0.7

            hvp_result = 1
            while hvp_result == 1:
                _, hvp_result = algorithms.run_rcig(coreset_images, coreset_labels, model_for_train.init, model_for_train.apply, ds_train, alg_config, key, inner_learning_rate, hvp_learning_rate, lr_tune = True)
                hvp_learning_rate *= 1.2

            hvp_learning_rate *= 0.7

        
    print("Done tuning learning rates")
    print(f'inner_learning_rate: {inner_learning_rate} hvp learning_rate: {hvp_learning_rate}')
    logging.info("Done tuning learning rates")
    logging.info(f'inner_learning_rate: {inner_learning_rate} hvp learning_rate: {hvp_learning_rate}')

    logging.info(f'Completed LR tune in {time.time() - start_time}s')


    #Training


    logging.info('Begin training')

    start_time = time.time()
    coreset_train_state, key, pool, inner_learning_rate, hvp_learning_rate = algorithms.run_rcig(coreset_images, coreset_labels, model_for_train.init, model_for_train.apply, ds_train, alg_config, key, inner_learning_rate, hvp_learning_rate, start_iter = 0)

    logging.info(f'Completed in {time.time() - start_time}s')

    logging.info(f'Saving final checkpoint')
    checkpoints.save_checkpoint(ckpt_dir = './{}/'.format(alg_config.output_dir), target = coreset_train_state, step = 'final', keep = 1e10)



    #Save version for visualizing (without ZCA transform)
    visualize_output_dict = {
        'coreset_images': np.array(rev_preprocess_op(coreset_train_state.ema_average['x_proto'])),
        'coreset_labels': np.array(coreset_train_state.ema_average['y_proto']),
        'dataset': config.dataset
    }

    if output_dir is not None:
        pickle.dump(visualize_output_dict, open('./{}/{}.pkl'.format(output_dir, 'distilled_dataset_vis'), 'wb'))

    print(f'new learning rates: {inner_learning_rate}, {hvp_learning_rate}')
    logging.info(f'new learning rates: {inner_learning_rate}, {hvp_learning_rate}')

    
if __name__ == '__main__':
    tf.config.experimental.set_visible_devices([], 'GPU')
    fire.Fire(main)
