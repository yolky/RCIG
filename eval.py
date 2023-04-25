import sys

# sys.path.append("..")

import os
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


from dataloader import get_dataset, configure_dataloader

# from lib.dataset.dataloader import get_dataset, configure_dataloader
# from lib.models.utils import create_model
# from lib.datadistillation.utils import save_dnfr_image, save_proto_np
# from lib.datadistillation.frepo import proto_train_and_evaluate, init_proto, ProtoHolder
# from lib.training.utils import create_train_state
# from lib.dataset.augmax import get_aug_by_name

from clu import metric_writers

from collections import namedtuple

# from jax.config import config as fsf
# fsf.update("jax_enable_x64", True)


from models import ResNet18, Conv, AlexNet, VGG11
from augmax import get_aug_by_name

import numpy as np
import jax.numpy as jnp
import algorithms
import optax
import time
import pickle

from flax.training import train_state, checkpoints

import json


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


def main(dataset_name = 'cifar10', data_path=None, zca_path=None, train_log=None, train_img=None, width=128, depth=3, normalization='identity', eval_lr = 0.0001, random_seed=0, message = 'eval_log', output_dir = None, max_cycles = 1000, config_path = None, checkpoint_path = None, save_name = 'eval_result', log_dir = None, eval_arch = 'conv', models_to_test = 5):
    # --------------------------------------
    # Setup
    # --------------------------------------


    if output_dir is None:
        output_dir = os.path.dirname(checkpoint_path)

    if log_dir is None:
        log_dir = output_dir

    logging.use_absl_handler()

    logging.get_absl_handler().use_absl_log_file('{}, {}'.format(int(time.time()), message), './{}/'.format(log_dir))
    absl.flags.FLAGS.mark_as_parsed() 
    logging.set_verbosity('info')
    
    logging.info('\n\n\n{}\n\n\n'.format(message))
    
    config = get_config()
    config.random_seed = random_seed
    config.train_log = train_log if train_log else 'train_log'
    config.train_img = train_img if train_img else 'train_img'


    config.dataset.data_path = data_path if data_path else 'data/tensorflow_datasets'
    config.dataset.zca_path = zca_path if zca_path else 'data/zca'
    config.dataset.name = dataset_name

    (ds_train, ds_test), preprocess_op, rev_preprocess_op, proto_scale = get_dataset(config.dataset)
    
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


    loaded_checkpoint = checkpoints.restore_checkpoint(f'./{checkpoint_path}', None)
    coreset_images = loaded_checkpoint['ema_average']['x_proto']
    coreset_labels = loaded_checkpoint['ema_average']['y_proto']
    

    if eval_arch == 'conv':
        model = Conv(use_softplus = False, beta = 20., num_classes = num_classes, width = width, depth = depth, normalization = normalization)
    elif eval_arch == 'resnet':
        model = ResNet18(output='logit', num_classes=num_classes, pooling='avg', normalization = normalization)
    elif eval_arch == 'vgg':
        model = VGG11(output='logit', num_classes=num_classes, pooling='avg', normalization = normalization)
    elif eval_arch == 'alexnet':
        model = AlexNet(output='logit', num_classes=num_classes, pooling='avg')

    use_batchnorm = normalization != 'identity'

    net_forward_init, net_forward_apply = model.init, model.apply

    key = jax.random.PRNGKey(random_seed)
    
    alg_config = ml_collections.ConfigDict()

    if config_path is not None:
        print(f'loading config from {config_path}')
        logging.info(f'loading config from {config_path}')
        loaded_dict = json.loads(open('./{}'.format(config_path), 'rb').read())
        loaded_dict['direct_batch_sizes'] = tuple(loaded_dict['direct_batch_sizes'])
        alg_config = ml_collections.config_dict.ConfigDict(loaded_dict)
    
    print(alg_config)
    logging.info(alg_config)

    if output_dir is not None:
        if not os.path.exists('./{}'.format(output_dir)):
            os.makedirs('./{}'.format(output_dir))

        with open('./{}/config.txt'.format(output_dir), 'a') as config_file:
            config_file.write(repr(alg_config))


    key, valid_key = jax.random.split(key)
    valid_keys = jax.random.split(valid_key, models_to_test)

    batch_size = 256 if coreset_images.shape[0] > 256 else None
    
    aug = get_aug_by_name(alg_config.test_aug, config.dataset.img_shape[0])

    eval_l2 = 0.00

    num_online_eval_updates = 1000 if coreset_images.shape[0] == 10 else 2000
    warmup_steps = 500

    learning_rate = eval_lr
    warmup_fn = optax.linear_schedule(init_value=0., end_value=learning_rate, transition_steps=warmup_steps)
    cosine_fn = optax.cosine_decay_schedule(init_value=learning_rate, alpha=0.01,
                                            decay_steps=max(num_online_eval_updates - warmup_steps, 1))
    learning_rate_fn = optax.join_schedules(schedules=[warmup_fn, cosine_fn], boundaries=[warmup_steps])

    if alg_config.use_flip:
        coreset_images = jnp.concatenate([coreset_images, jnp.flip(coreset_images, -2)], 0)
        coreset_labels = jnp.concatenate([coreset_labels, coreset_labels], 0 )
    

    logging.info('no data augmentation')

    acc_dict = {}

    accs = []
    for g in range(models_to_test):
        key, aug_key = jax.random.split(key)
        new_params = net_forward_init(valid_keys[g], coreset_images)
        if not use_batchnorm:
            bum = algorithms.TrainStateWithBatchStats.create(apply_fn = net_forward_apply, params = new_params['params'], tx = optax.chain(optax.adam(learning_rate_fn)), batch_stats = None, train_it = 0)
            for g in range(num_online_eval_updates//200):
                print(f'train checkpoint {(g) * 200} acc {algorithms.eval_on_test_set(bum, ds_test, has_bn = False, centering = False)}')

                bum, losses = algorithms.do_training_steps(bum, {'images': coreset_images, 'labels': coreset_labels}, aug_key, n_steps = 500, l2 = eval_l2, has_bn = False, train = False, batch_size = batch_size, max_batch_size = coreset_images.shape[0])
            accs.append(algorithms.eval_on_test_set(bum, ds_test, has_bn = False, centering = False))
        else:
            bum = algorithms.TrainStateWithBatchStats.create(apply_fn = net_forward_apply, params = new_params['params'], tx = optax.chain(optax.adam(learning_rate_fn)), batch_stats = new_params['batch_stats'], train_it = 0)
            for g in range(num_online_eval_updates//200):
                print(f'train checkpoint {(g) * 200} acc {algorithms.eval_on_test_set(bum, ds_test, has_bn = True, centering = False)}')
                bum, losses = algorithms.do_training_steps(bum, {'images': coreset_images, 'labels': coreset_labels}, aug_key, n_steps = 500, l2 = eval_l2, has_bn = True, train = True, batch_size = batch_size, max_batch_size = coreset_images.shape[0])

            accs.append(algorithms.eval_on_test_set(bum, ds_test, has_bn = True, centering = False))


        print(accs)
        
    logging.info('no data augmentation avg: {:.2f} pm {:.2f}'.format(100 * np.mean(accs), 100 * np.std(accs)))
    print('no data augmentation avg: {:.2f} pm {:.2f}'.format(100 * np.mean(accs), 100 * np.std(accs)))

    acc_dict['no_DA'] = np.array(accs)
        
    accs = []
    
    logging.info('with data augmentation')
    for g in range(models_to_test):
        key, aug_key = jax.random.split(key)
        new_params = net_forward_init(valid_keys[g], coreset_images)

        if not use_batchnorm:
            bum = algorithms.TrainStateWithBatchStats.create(apply_fn = net_forward_apply, params = new_params['params'], tx = optax.chain(optax.adam(learning_rate_fn)), batch_stats = None, train_it = 0)
            for g in range(num_online_eval_updates//500):
                print(f'train checkpoint {(g) * 500} acc {algorithms.eval_on_test_set(bum, ds_test, has_bn = False, centering = False)}')
                bum, losses = algorithms.do_training_steps(bum, {'images': coreset_images, 'labels': coreset_labels}, aug_key, n_steps = 500, l2 = eval_l2, has_bn = False, train = False, aug = aug, batch_size = batch_size, max_batch_size = coreset_images.shape[0])
            accs.append(algorithms.eval_on_test_set(bum, ds_test, has_bn = False, centering = False))

        else:
            bum = algorithms.TrainStateWithBatchStats.create(apply_fn = net_forward_apply, params = new_params['params'], tx = optax.chain(optax.adam(learning_rate_fn)), batch_stats = new_params['batch_stats'], train_it = 0)
            for g in range(num_online_eval_updates//500):
                print(f'train checkpoint {(g) * 500} acc {algorithms.eval_on_test_set(bum, ds_test, has_bn = True, centering = False)}')
                bum, losses = algorithms.do_training_steps(bum, {'images': coreset_images, 'labels': coreset_labels}, aug_key, n_steps = 500, l2 = eval_l2, has_bn = True, train = True, aug = aug, batch_size = batch_size, max_batch_size = coreset_images.shape[0])
            accs.append(algorithms.eval_on_test_set(bum, ds_test, has_bn = True, centering = False))

        print(accs)

    logging.info('with data augmentation avg: {:.2f} pm {:.2f}'.format(100 * np.mean(accs), 100 * np.std(accs)))
    print('with data augmentation avg: {:.2f} pm {:.2f}'.format(100 * np.mean(accs), 100 * np.std(accs)))

    acc_dict['DA'] = np.array(accs)


    

    if output_dir is not None:
        pickle.dump(acc_dict, open('./{}/{}.pkl'.format(output_dir, save_name), 'wb'))
    
    
if __name__ == '__main__':
    tf.config.experimental.set_visible_devices([], 'GPU')
    fire.Fire(main)