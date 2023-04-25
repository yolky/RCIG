# import eqm_prop_crap
import torch


import jax
import jax.numpy as jnp
import numpy as np
from flax.training import train_state, checkpoints
import ml_collections
import flax.linen as nn
from typing import Any, Callable, Sequence, Tuple
import jax.scipy as jsp
import functools
import flax
import optax
import utils
import time
from augmax import get_aug_by_name
import copy
from absl import logging

def identity(key, x):
    return x

def get_tree_mask(model_depth = 3, has_bn = False, learn_final = False):
    mask = {
        'Dense_0' : {
            'bias': True,
            'kernel': learn_final
        },
    }

    for i in range(model_depth):
        mask['Conv_{}'.format(i)] = True
        if has_bn:
            mask['BatchNorm_{}'.format(i)] = True

    return mask

@functools.partial(jax.jit, static_argnames=('pool_learning_rate', 'model_depth', 'has_bn', 'linearize', 'net_forward_apply', 'net_forward_init', 'img_shape', 'naive_loss'))
def get_new_train_state(key_inner, pool_learning_rate, model_depth, has_bn, linearize, net_forward_apply, net_forward_init, img_shape, naive_loss = False):
    inner_opt = optax.chain(
        optax.masked(optax.adam(learning_rate=0.0001), {'base_params': False,  'tangent_params': get_tree_mask(model_depth = model_depth, has_bn = has_bn, learn_final = naive_loss)}),
        optax.masked(optax.adam(learning_rate=pool_learning_rate), {'base_params': True,  'tangent_params': False}))

    new_params = net_forward_init(key_inner, jnp.zeros(shape = img_shape))
    
    key_inner = jax.random.split(key_inner)[0]
    new_batch_stats = new_params['batch_stats'] if has_bn else None
    new_params = new_params.unfreeze()
    
    if linearize:
        forward_linear = get_linear_forward(net_forward_apply, has_bn = has_bn)

        params_dict = {'base_params': new_params['params'], 'tangent_params': utils._zero_like(new_params['params'])}
        
        new_train_state = TrainStateWithBatchStats.create(apply_fn = forward_linear, params = params_dict, tx = inner_opt, batch_stats = new_batch_stats, train_it = 0)
    else:           
        if naive_loss: 
            new_train_state = TrainStateWithBatchStats.create(apply_fn = net_forward_apply, params = new_params['params'], tx = optax.adam(learning_rate = 0.0001), batch_stats = new_batch_stats, train_it = 0)
        else:
            forward_linear = get_linear_forward(net_forward_apply, has_bn = has_bn, linearize = False)

            params_dict = {'base_params': new_params['params'], 'tangent_params': utils._zero_like(new_params['params'])}
            
            new_train_state = TrainStateWithBatchStats.create(apply_fn = forward_linear, params = params_dict, tx = inner_opt, batch_stats = new_batch_stats, train_it = 0)
    
    return new_train_state, key_inner

def run_rcig(coreset_images_init, coreset_labels_init, net_forward_init, net_forward_apply, train_loader, alg_config, key, inner_learning_rate, hvp_learning_rate, test_fn = None, coreset_train_state = None, pool = None, lr_tune = False, start_iter = 0):
    if lr_tune:
        #We change some stuff in the config if we are tuning the inner/hessian inverse learning rates
        alg_config = copy.deepcopy(alg_config)
        alg_config.pool_model_count = 1
        alg_config.max_steps = 1
        alg_config.monitor_losses = True
        alg_config.aug = None
        alg_config.aug_repeats = 0

    alg_config = ml_collections.FrozenConfigDict(alg_config)
    

    #Instatiate model pool
    if pool is None:
        model_pool = []
        for m in range(alg_config.pool_model_count):
            new_train_state, key = get_new_train_state(key, alg_config.pool_learning_rate, alg_config.model_depth, alg_config.has_bn, alg_config.linearize, net_forward_apply, net_forward_init, coreset_images_init.shape, naive_loss = alg_config.naive_loss)
            model_pool.append(jax.device_put(new_train_state, jax.devices('cpu')[0]))
    else:
        model_pool = pool
        
    if coreset_train_state is None:
        proto_obj = ProtoHolder(coreset_images_init, coreset_labels_init, 0.0, coreset_images_init.shape[0], learn_label=alg_config.learn_labels, use_flip = alg_config.use_flip)
        lr_schedule = alg_config.proto_learning_rate
        coreset_opt = optax.chain(
                optax.masked(optax.adabelief(learning_rate=lr_schedule), {'x_proto': True,  'y_proto': True, 'log_temp': False}),
                optax.masked(optax.adabelief(learning_rate=0.03), {'x_proto': False,  'y_proto': False, 'log_temp': True}),
        )
        coreset_init_params = proto_obj.init({'params': key}).unfreeze()['params']
        coreset_train_state = CoresetTrainState.create(apply_fn = proto_obj.apply, tx = coreset_opt, params = coreset_init_params, train_it = 0, ema_average = coreset_init_params, ema_hidden = utils._zero_like(coreset_init_params))        
    
    n_steps = 0
    
    aug = identity if alg_config.aug is None else get_aug_by_name(alg_config.aug, alg_config.img_size)
    
    print(alg_config.aug)

    inner_lr_stats = []
    hvp_lr_stats = []
    lr_monitor_interval = 50

    start_time = time.time()

    if start_iter == 0 and not lr_tune and alg_config.output_dir is not None:
        logging.info(f'Saving checkpoint at iter 0 at time {time.time() - start_time}')
        checkpoints.save_checkpoint(ckpt_dir = './{}/'.format(alg_config.output_dir), target = coreset_train_state, step = 0, keep = 1e10)
    
    while(n_steps < alg_config.max_steps):
        for train_images, train_labels in train_loader.as_numpy_iterator():
            debug_info = [n_steps]
            model_index = jax.random.randint(key, (), 0, alg_config.pool_model_count)

            key = jax.random.split(key)[0]

            
            #Select Model state

            selected_model_state = jax.device_put(model_pool[model_index], jax.devices('gpu')[0])



            #Do inner steps

            n_steps_to_opt = alg_config.n_inner_steps

            key, train_key = jax.random.split(key)
            
            x_proto, y_proto, _ = coreset_train_state.apply_fn({'params': coreset_train_state.params})
            coreset_batch = {'images': x_proto, 'labels': y_proto}            

            new_train_state, losses = do_training_steps(selected_model_state, coreset_batch, train_key, n_steps = n_steps_to_opt, l2 = alg_config.l2, has_bn = alg_config.has_bn, train = False, aug = aug, do_krr = not alg_config.naive_loss, batch_size = alg_config.inner_train_batch_size, inject_lr = inner_learning_rate, alg_config = alg_config)


            if alg_config.monitor_losses:
                print(losses[:20])
    



            #This is all stuff for making sure our learning rates aren't too high. TL;DR if we notice that the inner losses are often diverging, we decrease the learning rate, and we slightly increase if they are all monotonically decreasing

            loss_diag = (losses[:alg_config.n_inner_steps] - jnp.roll(losses[:alg_config.n_inner_steps], -1))
            if alg_config.n_inner_steps == 0:
                inner_lr_stats.append(0)
            elif jnp.all(loss_diag[:-1] > 0):
                # print("monotonic")
                inner_lr_stats.append(1)
            elif jnp.all(loss_diag[-1] > 0):
                # print('fail')
                inner_lr_stats.append(-1)
                # print("HALVING")
            else:
                # print('unstable')
                inner_lr_stats.append(0)

            debug_info.append(inner_learning_rate)
            debug_info.append(inner_lr_stats[-1])
            debug_info.append(float(losses[0]))
            debug_info.append(float(losses[alg_config.n_inner_steps - 1]))
            if alg_config.n_inner_steps > 0:
                debug_info.append(float(losses[alg_config.n_inner_steps - 1])/float(losses[0]))
            else:
                debug_info.append(0)

            if n_steps%lr_monitor_interval == 0 and n_steps > 0:
                if jnp.mean(np.array(inner_lr_stats) == -1) >= .29:
                    # print("FAILING DECREASE")
                    #more than 30% fail, shoudl decrease learning rate
                    inner_learning_rate *= 0.9

                elif jnp.mean(np.array(inner_lr_stats) == 1) >= .7:
                    #more than 95% monotonic decrease, can increase the learning rate
                    # print("STABLE INCREASING")
                    inner_learning_rate *= 1.05

                inner_lr_stats = []



            #Comput the implicit gradient
            
            key, meta_train_key = jax.random.split(key)
            coreset_train_state, (outer_loss, outer_acc, residuals, grad_norm, update_norms, update_maxes, norm_ratios) = do_meta_train_step(new_train_state, coreset_train_state, train_images, train_labels, has_bn = alg_config.has_bn, l2 = alg_config.l2, 
                n_hinv_steps = alg_config.n_hinv_steps, do_krr = not alg_config.naive_loss, aug = aug, aug_repeats = alg_config.aug_repeats, aug_key = meta_train_key,
                lr = hvp_learning_rate, normal_repeats = alg_config.normal_repeats,
                direct_batch_sizes = alg_config.direct_batch_sizes, implicit_batch_size = alg_config.implicit_batch_size,
                hinv_batch_size = alg_config.hinv_batch_size,  do_precompute = alg_config.do_precompute, max_forward_batch_size = alg_config.max_forward_batch_size, alg_config = alg_config)
            
            if alg_config.monitor_losses:
                print(residuals[:20])

            

            #Again more logging/debug stuff for making sure our hessian inverse computation isn't diverging
            loss_diag = (residuals[:alg_config.n_hinv_steps] - jnp.roll(residuals[:alg_config.n_hinv_steps], -1))
            if alg_config.n_hinv_steps == 0:
                hvp_lr_stats.append(0)
            elif jnp.all(loss_diag[:-1] > 0):
                # print("monotonic")
                hvp_lr_stats.append(1)
            elif jnp.all(loss_diag[-1] > 0):
                # print('fail')
                hvp_lr_stats.append(-1)
                # print("HALVING")
            else:
                # print('unstable')
                hvp_lr_stats.append(0)


            debug_info.append(hvp_learning_rate)
            debug_info.append(hvp_lr_stats[-1])
            debug_info.append(grad_norm)
            debug_info.append(update_norms)
            debug_info.append(update_maxes)
            debug_info.append(norm_ratios)




            print(f'iter: {n_steps + 1}, outer loss: {outer_loss}, outer acc: {outer_acc}')

            debug_info.append(float(outer_loss))
            debug_info.append(float(outer_acc))
        
            logging.info(debug_info)



            #Do training steps for the pool models
            
            key, outer_train_key = jax.random.split(key)
            n_steps_to_opt_pool = jax.random.randint(key, (), 1, alg_config.n_max_steps_pool)

            x_proto, y_proto, _ = coreset_train_state.apply_fn({'params': coreset_train_state.params})
            coreset_batch = {'images': x_proto, 'labels': y_proto}

            model_pool[model_index], outer_loss = do_training_steps(selected_model_state, coreset_batch, outer_train_key, n_steps = n_steps_to_opt_pool, has_bn = alg_config.has_bn, use_base_params = not alg_config.naive_loss, aug = aug, batch_size = alg_config.pool_train_batch_size, max_batch_size = coreset_train_state.params['x_proto'].shape[0], train = True)

            model_pool[model_index] = jax.device_put(model_pool[model_index], jax.devices('cpu')[0])
            

            #create new pool model if it has done too many training steps
            if model_pool[model_index].train_it >= alg_config.max_online_steps:
                new_train_state, key = get_new_train_state(key, alg_config.pool_learning_rate, alg_config.model_depth, alg_config.has_bn, alg_config.linearize, net_forward_apply, net_forward_init, coreset_images_init.shape, naive_loss = alg_config.naive_loss)
                model_pool[model_index] = jax.device_put(new_train_state, jax.devices('cpu')[0])

            n_steps += 1


            #checkpoint saving
            if start_iter + n_steps in alg_config.checkpoint_iters and not lr_tune and n_steps != 0 and alg_config.output_dir is not None:
                print(f"saving at iter {n_steps + start_iter} at time {time.time() - start_time}")
                logging.info(f'Saving checkpoint at iter {n_steps + start_iter} at time {time.time() - start_time}')
                checkpoints.save_checkpoint(ckpt_dir = './{}/'.format(alg_config.output_dir), target = coreset_train_state, step = n_steps + start_iter, keep = 1e10)

            
            if n_steps >= alg_config.max_steps:
                break
                
    # return coreset_train_state.params['x_proto']
    if lr_tune:
        return inner_lr_stats[0], hvp_lr_stats[0]

    return coreset_train_state, key, model_pool, inner_learning_rate, hvp_learning_rate

@functools.partial(jax.jit, static_argnames=('has_bn', 'do_krr', 'aug', 'aug_repeats', 'batch_sizes', 'normal_repeats', 'max_forward_batch_size'))
def get_gt_and_direct(model_train_state, coreset_train_state, train_images, train_labels, has_bn = False, l2 = 0., n_hinv_steps = 20, cg_init = None, do_krr = False, aug = None, aug_repeats = 0, aug_key = None, normal_repeats = 1, batch_sizes = None, pre_s = None, pre_t = None, pre_s_aug = None, max_forward_batch_size = None):
    if has_bn:
        batch_stats = model_train_state.batch_stats
    else:
        batch_stats = None


    if not do_krr:
        (loss, (_, acc, _)), g_t = jax.value_and_grad(get_training_loss_l2, argnums = 0, has_aux = True)(model_train_state.params, train_images, train_labels, model_train_state, l2 = 0, train = False, has_bn = has_bn, batch_stats = batch_stats)
        direct_grad = utils._zero_like(coreset_train_state.params)
    else:
        @functools.partial(jax.jit, static_argnames=('batch_aug'))
        def body_fn(i, val, batch_aug = identity):
            g_t_cum, direct_grad_cum, loss, acc, key, pre_s_inner = val

            key, aug_key, grad_key1, grad_key2 = jax.random.split(key, 4)


            #This (and any mention about grad indices) has to do with randomly sampling samples to backpropagate through. It's only relevant for bigger coreset sizes
            if batch_sizes[0] is not None:
                grad_indices1 = jax.random.choice(grad_key1, coreset_train_state.apply_fn({'params': coreset_train_state.params})[0].shape[0], shape = [batch_sizes[0]], replace = False)
            else:
                grad_indices1 = None
                
            if batch_sizes[1] is not None:
                grad_indices2 = jax.random.choice(grad_key2, train_images.shape[0], shape = [batch_sizes[1]], replace = False)
            else:
                grad_indices2 = None
            
            (loss, (acc, _)), (g_t, direct_grad) = jax.value_and_grad(get_krr_loss, argnums = (0,1), has_aux = True)(model_train_state.params, coreset_train_state.params, coreset_train_state.apply_fn, train_images, train_labels, model_train_state.apply_fn, has_bn = has_bn, batch_stats = batch_stats, l2 = l2, grad_indices1 = grad_indices1, grad_indices2 = grad_indices2, aug = batch_aug, pre_s = pre_s_inner, pre_t = pre_t, max_forward_batch_size = max_forward_batch_size)    

            g_t_cum = utils._add(g_t_cum, g_t)
            direct_grad_cum = utils._add(direct_grad_cum, direct_grad)

            return (g_t_cum, direct_grad_cum, loss, acc, key, pre_s_inner)
            

        g_t, direct_grad, loss, acc, aug_key, _ = jax.lax.fori_loop(0, normal_repeats, body_fn, (utils._sub(model_train_state.params, model_train_state.params), utils._sub(coreset_train_state.params, coreset_train_state.params), 0,0, aug_key, pre_s))
        g_t_aug, direct_grad_aug, _, _, aug_key, _ = jax.lax.fori_loop(0, aug_repeats, utils.bind(body_fn, ...,..., aug), (utils._sub(model_train_state.params, model_train_state.params), utils._sub(coreset_train_state.params, coreset_train_state.params), 0,0, aug_key, pre_s_aug))

        direct_grad = utils._add(direct_grad, direct_grad_aug)
        g_t = utils._add(g_t, g_t_aug)

        direct_grad = multiply_by_scalar(direct_grad, 1/(aug_repeats + normal_repeats))
        g_t = multiply_by_scalar(g_t, 1/(aug_repeats + normal_repeats))


    return g_t, direct_grad, aug_key, loss, acc

@functools.partial(jax.jit, static_argnames=('has_bn', 'do_krr', 'aug', 'aug_repeats', 'normal_repeats', 'batch_size', 'use_x64'))
def get_implicit_grad(h_inv_vp, model_train_state, coreset_train_state, train_images, train_labels, has_bn = False, l2 = 0., n_hinv_steps = 20, cg_init = None, do_krr = False, aug = None, aug_repeats = 0, aug_key = None, normal_repeats = 1, batch_size = None, pre_s = None, pre_s_aug = None, use_x64 = False):
    if has_bn:
        batch_stats = model_train_state.batch_stats
    else:
        batch_stats = None

    def cg_shit(coreset_train_state_params_inner, aug_inner, key_inner, pre_s_inner):
        # h_inv_vp, hists = get_v_h_inv_approx(params, coreset_images, coreset_labels, net_forward_apply, g_t, n_iters = 20)
        
        images_inner, labels_inner, log_temp = coreset_train_state.apply_fn({'params': coreset_train_state_params_inner})

        aug_key, index_key = jax.random.split(key_inner)
        aug_images = aug_inner(aug_key, images_inner)

        if batch_size is not None:
            grad_indices = jax.random.choice(index_key, aug_images.shape[0], shape = [batch_size], replace = False)
        else:
            grad_indices = None

        if not do_krr:
            g_s = jax.grad(get_training_loss_l2, argnums = 0, has_aux = True)(model_train_state.params, aug_images, labels_inner, model_train_state, l2 = l2, has_bn = has_bn, batch_stats = model_train_state.batch_stats)[0]
            return -1 * get_dot_product(h_inv_vp, g_s), g_s    
        else:
            g_s = jax.grad(get_krr_loss_gd, argnums = 0, has_aux = True)(model_train_state.params, model_train_state, aug_images, labels_inner, None, None, l2 = l2, has_bn = has_bn, batch_stats = batch_stats, self_loss = True, log_temp = log_temp, grad_indices = grad_indices, pre_s = pre_s_inner, use_x64 = use_x64)[0]
            return -1 * get_dot_product(h_inv_vp, g_s), g_s
        
        
    @functools.partial(jax.jit, static_argnames=('batch_aug'))
    def body_fn(i, val, batch_aug = identity):
        implicit_grad_cum, key, pre_s_inner = val

        implicit_grad, _ = jax.grad(cg_shit, argnums = 0, has_aux = True)(coreset_train_state.params, batch_aug, key, pre_s_inner)
        implicit_grad_cum = utils._add(implicit_grad_cum, implicit_grad)

        key = jax.random.split(key, 2)[0]

        return (implicit_grad_cum, key, pre_s_inner)


    implicit_grad, aug_key,_ = jax.lax.fori_loop(0, normal_repeats, body_fn, (utils._sub(coreset_train_state.params, coreset_train_state.params), aug_key, pre_s))
    implicit_grad_aug, aug_key, _ = jax.lax.fori_loop(0, aug_repeats, utils.bind(body_fn, ...,..., aug), (utils._sub(coreset_train_state.params, coreset_train_state.params), aug_key, pre_s_aug))

    implicit_grad = utils._add(implicit_grad, implicit_grad_aug)
    implicit_grad = multiply_by_scalar(implicit_grad, 1/(aug_repeats + normal_repeats))

    return implicit_grad

@functools.partial(jax.jit, static_argnames=('has_bn', 'do_krr', 'aug', 'aug_repeats', 'direct_batch_sizes', 'implicit_batch_size', 'normal_repeats', 'do_precompute', 'hinv_batch_size', 'max_forward_batch_size', 'alg_config'))
def do_meta_train_step(model_train_state, coreset_train_state, train_images, train_labels, has_bn = False, l2 = 0., n_hinv_steps = 20, cg_init = None, do_krr = False, aug = None, aug_repeats = 0, aug_key = None, lr = 3., normal_repeats = 1, direct_batch_sizes = None, implicit_batch_size = None, do_precompute = False, hinv_batch_size = None, max_forward_batch_size = None, alg_config = None):

    if has_bn:
        batch_stats = model_train_state.batch_stats
    else:
        batch_stats = None

    if do_precompute:
        pre_s, pre_t = jax.lax.stop_gradient(get_krr_loss(model_train_state.params, coreset_train_state.params, coreset_train_state.apply_fn, train_images, train_labels, model_train_state.apply_fn, has_bn = has_bn, batch_stats = batch_stats, l2 = l2, aug = identity, do_precompute = True, max_forward_batch_size = max_forward_batch_size, use_x64 = alg_config.use_x64))
        aug_key, precompute_key = jax.random.split(aug_key)
        pre_s_aug, _ = jax.lax.stop_gradient(get_krr_loss(model_train_state.params, coreset_train_state.params, coreset_train_state.apply_fn, train_images, train_labels, model_train_state.apply_fn, has_bn = has_bn, batch_stats = batch_stats, l2 = l2, aug = aug, do_precompute = True, aug_key = precompute_key, max_forward_batch_size = max_forward_batch_size, use_x64 = alg_config.use_x64))
    else:
        pre_s, pre_t, pre_s_aug = None, None, None

    g_t, direct_grad, aug_key, loss, acc = jax.lax.stop_gradient(get_gt_and_direct(model_train_state, coreset_train_state, train_images, train_labels, has_bn = has_bn, l2 = l2, n_hinv_steps = n_hinv_steps, cg_init = cg_init, do_krr = do_krr, aug = aug, aug_repeats = aug_repeats, aug_key = aug_key, normal_repeats = normal_repeats, batch_sizes = direct_batch_sizes, pre_s = pre_s, pre_s_aug = pre_s_aug, pre_t = pre_t, max_forward_batch_size = max_forward_batch_size))


    if aug is not None and do_precompute:
        h_inv_vp, residuals = jax.lax.stop_gradient(get_h_inv_vp(coreset_train_state, model_train_state, g_t, n_steps = n_hinv_steps, l2 = l2, has_bn = has_bn, batch_stats = batch_stats, init = cg_init, do_krr = do_krr, lr = lr, pre_s = pre_s_aug, batch_size = hinv_batch_size, alg_config = alg_config))
    else:
        h_inv_vp, residuals = jax.lax.stop_gradient(get_h_inv_vp(coreset_train_state, model_train_state, g_t, n_steps = n_hinv_steps, l2 = l2, has_bn = has_bn, batch_stats = batch_stats, init = cg_init, do_krr = do_krr, lr = lr, pre_s = pre_s, batch_size = hinv_batch_size, alg_config = alg_config))

    
    implicit_grad = get_implicit_grad(h_inv_vp, model_train_state, coreset_train_state, train_images, train_labels, has_bn = has_bn, l2 = l2, n_hinv_steps = n_hinv_steps, cg_init = cg_init, do_krr = do_krr, aug = aug, aug_repeats = aug_repeats, aug_key = aug_key, normal_repeats = normal_repeats, batch_size = implicit_batch_size, pre_s = pre_s, pre_s_aug = pre_s_aug, use_x64 = alg_config.use_x64)


    #clip implicit gradient norm so it isn't larger than the direct gradient norm
    #we found that this helps stability for high resolution datasets, as the implicit gradient could grow very very large
    clip = not alg_config.naive_loss

    if clip:
        igrad_norm = jax.tree_map(jnp.linalg.norm, implicit_grad)
        dgrad_norm = jax.tree_map(jnp.linalg.norm, direct_grad)
        max_norm_tree = jax.tree_map(jnp.minimum, dgrad_norm, igrad_norm)
        implicit_grad = jax.tree_map(lambda g, g_norm, max_norm: (g/g_norm) * max_norm, implicit_grad, igrad_norm, max_norm_tree)
        implicit_grad = jax.tree_map(jnp.nan_to_num, implicit_grad)

    norm_ratios = utils._divide(jax.tree_map(jnp.linalg.norm, direct_grad), jax.tree_map(jnp.linalg.norm, implicit_grad))

    grad = utils._add(direct_grad, implicit_grad)

    
    coreset_train_state, updates = coreset_train_state.apply_gradients_get_updates(grads = grad, train_it = coreset_train_state.train_it + 1)

    new_ema_hidden, new_ema_average = get_updated_ema(coreset_train_state.params, coreset_train_state.ema_hidden, 0.99, coreset_train_state.train_it, order = 1)
    coreset_train_state = coreset_train_state.replace(ema_average = new_ema_average, ema_hidden = new_ema_hidden)

    
    return coreset_train_state, (loss, acc, residuals, jax.tree_map(jnp.linalg.norm, grad), jax.tree_map(jnp.linalg.norm, updates), jax.tree_map(jnp.max, jax.tree_map(jnp.abs, updates)), norm_ratios)

@functools.partial(jax.jit, static_argnames=('has_bn', 'other_opt', 'do_krr', 'aug', 'batch_size', 'alg_config'))
def get_h_inv_vp(coreset_train_state, model_train_state, g_t, n_steps = 20, l2 = 0.0, has_bn = False, batch_stats = None, other_opt = False, init = None, do_krr = False, aug = identity, aug_key = jax.random.PRNGKey(0), lr = 3., pre_s = None, batch_size = None, alg_config = None):
    opt_init, opt_update = optax.chain(optax.adam(lr))
    residuals = jnp.zeros(1000)
    
    if init is None:
        x = utils._zero_like(g_t)
    else:
        x = init
    
    opt_state = opt_init(x)

    def body_fn(i, val):
        x, opt_state, residuals, aug_key = val
        x_proto, y_proto, _ = coreset_train_state.apply_fn({'params': coreset_train_state.params})

        if not do_krr:
            Hx = get_training_loss_hvp(model_train_state.params, x_proto, y_proto, model_train_state, x, l2 = l2, has_bn = has_bn, batch_stats = model_train_state.batch_stats)
        else:
            aug_key, grad_key = jax.random.split(aug_key)
            aug_images = aug(aug_key, x_proto)
            if batch_size is None:
                grad_indices = None
            else:
                grad_indices = jax.random.choice(grad_key, aug_images.shape[0], shape = [batch_size], replace = False)
            Hx = get_training_loss_hvp_krr(model_train_state.params, aug_images, y_proto, model_train_state, x, l2 = l2, has_bn = has_bn, batch_stats = model_train_state.batch_stats, grad_indices = grad_indices, pre_s = pre_s, alg_config = alg_config)

        grad = utils._sub(Hx, g_t)

        residual = get_dot_product(x, utils._sub(Hx, multiply_by_scalar(g_t, 2)))

        updates, new_opt_state = opt_update(grad, opt_state, x)
        x = optax.apply_updates(x, updates)


        residuals = residuals.at[i].set(residual)

        return x, new_opt_state, residuals, aug_key

    h_inv_vp, _, residuals, _  = jax.lax.fori_loop(0, n_steps, body_fn, (x, opt_state, residuals, aug_key))

    return h_inv_vp, residuals

def invert_grad_indices(grad_indices, max_size):
    grad_mask = jnp.zeros(shape = [max_size], dtype = jnp.bool_).at[grad_indices].set(True)
    return jnp.nonzero(~grad_mask, size = max_size - grad_indices.shape[0])

@functools.partial(jax.jit, static_argnames=('train', 'has_bn', 'net_forward_apply', 'coreset_train_state_apply', 'aug', 'do_precompute', 'max_forward_batch_size', 'use_x64'))
def get_krr_loss(params, coreset_train_state_params, coreset_train_state_apply, images2, labels2, net_forward_apply, has_bn = True, batch_stats = None, l2 = 0., aug = None, aug_key = jax.random.PRNGKey(0), grad_indices1 = None, grad_indices2 = None, do_precompute = False, pre_s = None, pre_t = None, max_forward_batch_size = None, use_x64 = False):
    
    images, labels, log_temp = coreset_train_state_apply({'params': coreset_train_state_params})

    if aug is not None:
        images = aug(aug_key, images)
    
    if has_bn:
        net_variables = {'params': params, 'batch_stats': batch_stats}
    else:
        net_variables = {'params': params}
    

    if do_precompute:
        feat_s, out_s = batch_precompute(net_forward_apply, net_variables, images, max_forward_batch_size = max_forward_batch_size)
    elif pre_s is not None and grad_indices1 is not None:
        inv_grad_indices = invert_grad_indices(grad_indices1, images.shape[0])
        (_, feat_s1), (out_s1, _), _ = net_forward_apply(net_variables, images[grad_indices1], features = True, train = False, mutable=['batch_stats'], return_all = True)
        feat_s2 = pre_s[0][inv_grad_indices]
        out_s2 = pre_s[1][inv_grad_indices]

        grad_labels = labels[grad_indices1]
        no_grad_labels = labels[inv_grad_indices]

        labels = jnp.concatenate([grad_labels, no_grad_labels])
        feat_s = jnp.concatenate([feat_s1, feat_s2])
        out_s = jnp.concatenate([out_s1, out_s2])

    elif grad_indices1 is not None:
        inv_grad_indices = invert_grad_indices(grad_indices1, images.shape[0])
        (_, feat_s1), (out_s1, _), _ = net_forward_apply(net_variables, images[grad_indices1], features = True, train = False, mutable=['batch_stats'], return_all = True)
        feat_s2, out_s2 = batch_precompute(net_forward_apply, net_variables, images[inv_grad_indices], max_forward_batch_size = max_forward_batch_size)

        grad_labels = labels[grad_indices1]
        no_grad_labels = labels[inv_grad_indices]

        labels = jnp.concatenate([grad_labels, no_grad_labels])
        feat_s = jnp.concatenate([feat_s1, feat_s2])
        out_s = jnp.concatenate([out_s1, out_s2])
    else:
        (_, feat_s), (out_s, _), _ = net_forward_apply(net_variables, images, features = True, train = False, mutable=['batch_stats'], return_all = True)


    if do_precompute:
        feat_t, out_t = batch_precompute(net_forward_apply, net_variables, images2, max_forward_batch_size = max_forward_batch_size)
    elif pre_t is not None and grad_indices2 is not None:
        inv_grad_indices = invert_grad_indices(grad_indices2, images2.shape[0])
        (_, feat_t1), (out_t1, _), _ = net_forward_apply(net_variables, images2[grad_indices2], features = True, train = False, mutable=['batch_stats'], return_all = True)

        feat_t2 = pre_t[0][inv_grad_indices]
        out_t2 = pre_t[1][inv_grad_indices]

        grad_labels2 = labels2[grad_indices2]
        no_grad_labels2 = labels2[inv_grad_indices]

        labels2 = jnp.concatenate([grad_labels2, no_grad_labels2])
        feat_t = jnp.concatenate([feat_t1, feat_t2])
        out_t = jnp.concatenate([out_t1, out_t2])
    elif grad_indices2 is not None:
        inv_grad_indices = invert_grad_indices(grad_indices2, images2.shape[0])
        (_, feat_t1), (out_t1, _), _ = net_forward_apply(net_variables, images2[grad_indices2], features = True, train = False, mutable=['batch_stats'], return_all = True)
        feat_t2, out_t2 = batch_precompute(net_forward_apply, net_variables, images2[inv_grad_indices], max_forward_batch_size = max_forward_batch_size)

        grad_labels2 = labels2[grad_indices2]
        no_grad_labels2 = labels2[inv_grad_indices]

        labels2 = jnp.concatenate([grad_labels2, no_grad_labels2])
        feat_t = jnp.concatenate([feat_t1, feat_t2])
        out_t = jnp.concatenate([out_t1, out_t2])
    else:
        (_, feat_t), (out_t, _), _ = net_forward_apply(net_variables, images2, features = True, train = False, mutable=['batch_stats'], return_all = True)

    if do_precompute:
        return (feat_s, out_s), (feat_t, out_t)

    
    if use_x64:
        K_ss = (feat_s @ feat_s.T).astype(jnp.float64)
    else:
        K_ss = feat_s @ feat_s.T
    K_ts = feat_t @ feat_s.T
    
    K_ss_reg = K_ss + l2 * jnp.eye(K_ss.shape[0])

    preds = out_t + K_ts @ jnp.linalg.solve(K_ss_reg, labels - out_s)
        
    y_hat = labels2
    
    acc = jnp.mean(preds.argmax(1) == labels2.argmax(1))
    
    loss = 0.5 * jnp.mean((preds - y_hat)**2)
    labels2_oh = labels2 - jnp.min(labels2)
    loss = jnp.mean(optax.softmax_cross_entropy(preds * jnp.exp(log_temp), labels2_oh))

    dim = labels.shape[-1]
    val, idx = jax.lax.top_k(labels, k=2)
    margin = jnp.minimum(val[:, 0] - val[:, 1], 1 /(2 * dim))
    #small loss so that the top label stays at least 1/(2c) higher than the next label
    #we are unsure if this actually does anything useful but we had it in the code when we ran the experiments
    #it is quite likely it makes no difference
            
    return loss.astype(jnp.float32) - margin.mean(), (acc, 0)

@functools.partial(jax.jit, static_argnames=('apply_fn', 'max_forward_batch_size'))
def batch_precompute(apply_fn, variables, images, max_forward_batch_size = None):
    if max_forward_batch_size is None or max_forward_batch_size >= images.shape[0]:
        (_, feat), (out, _), _ = jax.lax.stop_gradient(apply_fn(variables, images, features = True, train = False, mutable=['batch_stats'], return_all = True))

        return feat, out

    else:
        def body_fn(carry, t):
            i = carry

            batch_images = jax.lax.stop_gradient(jax.lax.dynamic_slice(images, (i * max_forward_batch_size, 0, 0, 0), (max_forward_batch_size, images.shape[1], images.shape[2], images.shape[3])))
            (_, feat), (out, _), _ = jax.lax.stop_gradient(apply_fn(variables, batch_images, features = True, train = False, mutable=['batch_stats'], return_all = True))

            return i+1, jax.lax.stop_gradient([feat, out])

        _, [feats, outs] = jax.lax.scan(body_fn, 0, jnp.arange((images.shape[0] - 1)//max_forward_batch_size))
        final_batch_size = ((images.shape[0] - 1) % max_forward_batch_size) + 1

        (_, feat), (out, _), _ = jax.lax.stop_gradient(apply_fn(variables, images[images.shape[0]-final_batch_size:], features = True, train = False, mutable=['batch_stats'], return_all = True))


        feats, outs =  jnp.concatenate([feats.reshape(-1, feats.shape[-1]), feat]), jnp.concatenate([outs.reshape(-1, outs.shape[-1]), out])

        return feats, outs

@functools.partial(jax.jit, static_argnames=('train', 'has_bn', 'self_loss', 'use_base_params', 'max_forward_batch_size', 'use_x64'))
def get_krr_loss_gd(params, net_train_state, images, labels, images2, labels2, has_bn = True, batch_stats = None, l2 = 0., self_loss = True, use_base_params = False, log_temp = 0., grad_indices = None, pre_s = None, max_forward_batch_size = None, use_x64 = False):
    if has_bn:
        net_variables = {'params': params, 'batch_stats': batch_stats}
    else:
        net_variables = {'params': params}

    
    if pre_s is not None and grad_indices is not None:
        inv_grad_indices = invert_grad_indices(grad_indices, images.shape[0])
        (_, feat_s1), (out_s1, _), _ = net_train_state.apply_fn(net_variables, images[grad_indices], features = True, train = False, mutable=['batch_stats'], return_all = True)
        
        feat_s2 = pre_s[0][inv_grad_indices]
        out_s2 = pre_s[1][inv_grad_indices]

        grad_labels = labels[grad_indices]
        no_grad_labels = labels[inv_grad_indices]

        labels = jnp.concatenate([grad_labels, no_grad_labels])
        feat_s = jnp.concatenate([feat_s1, feat_s2])
        out_s = jnp.concatenate([out_s1, out_s2])
    elif grad_indices is not None:
        inv_grad_indices = invert_grad_indices(grad_indices, images.shape[0])
        (_, feat_s1), (out_s1, _), _ = net_train_state.apply_fn(net_variables, images[grad_indices], features = True, train = False, mutable=['batch_stats'], return_all = True)

        feat_s2, out_s2 = batch_precompute(net_train_state.apply_fn, net_variables, images[inv_grad_indices], max_forward_batch_size = max_forward_batch_size)

        grad_labels = labels[grad_indices]
        no_grad_labels = labels[inv_grad_indices]

        labels = jnp.concatenate([grad_labels, no_grad_labels])
        feat_s = jnp.concatenate([feat_s1, feat_s2])
        out_s = jnp.concatenate([out_s1, out_s2])
    else:
        (_, feat_s), (out_s, _), _ = net_train_state.apply_fn(net_variables, images, features = True, train = False, mutable=['batch_stats'], return_all = True) 
    

    if use_x64:
        K_ss = (feat_s @ feat_s.T).astype(jnp.float64)
    else:
        K_ss = feat_s @ feat_s.T
    
    if not self_loss:
        (_, feat_t), (out_t, _), _ = net_train_state.apply_fn(net_variables, images2, features = True, train = False, mutable=['batch_stats'], return_all = True)
        K_ts = feat_t @ feat_s.T
    
    K_ss_reg = K_ss + l2 * jnp.eye(K_ss.shape[0])

    spectral_weights = (jnp.linalg.solve(K_ss_reg, labels - out_s))


    if self_loss:
        
        preds = out_s + K_ss @ spectral_weights
        wtw = (spectral_weights.T @ K_ss @ spectral_weights)

        self_err = labels - preds
        
        loss = 0.5 * jnp.trace(self_err @ self_err.T)

        added_body = params['tangent_params']

        weight_decay_loss =  0.5 * l2 * (jnp.trace(wtw) + get_dot_product(added_body, added_body))

        loss += weight_decay_loss

        loss = loss/(labels.shape[0] * labels.shape[1])

        return loss.astype(jnp.float32), (batch_stats, 1, None)

    else:
        preds = out_t + K_ts @ spectral_weights
        err = labels2 - preds
        loss = 0.5 * jnp.mean(err**2)

        labels2_oh = labels2 - jnp.min(labels2)

        loss = jnp.mean(optax.softmax_cross_entropy(preds * jnp.exp(log_temp), labels2_oh))

        acc = jnp.mean(preds.argmax(1) == labels2.argmax(1))

        return loss.astype(jnp.float32), (batch_stats, acc, None)

@functools.partial(jax.jit, static_argnames=('train', 'has_bn', 'use_base_params', 'centering'))
def get_training_loss_l2(params, images, labels, net_train_state, l2 = 0., train = False, has_bn = False, batch_stats = None, use_base_params = False, centering = False, init_params = None, init_batch_params = None):
    if has_bn:
        variables = {'params': params, 'batch_stats': batch_stats}
    else:
        variables = {'params': params}


    mutable = ['batch_stats'] if train else []

    if centering:
        if has_bn:
            init_variables = {'params': init_params, 'batch_stats': init_batch_params}
        else:
            init_variables = {'params': init_params}

    if use_base_params:
        outputs, new_batch_stats = net_train_state.apply_fn(variables, images, train = train, mutable=mutable, use_base_params = use_base_params)
        
    else:
        outputs, new_batch_stats = net_train_state.apply_fn(variables, images, train = train, mutable=mutable)
        if centering:
            outputs_init, _ = net_train_state.apply_fn(init_variables, images, train = train, mutable=mutable)

            outputs = outputs - outputs_init

    loss = jnp.sum(0.5 * (outputs - labels)**2)
    
    if type(l2) is dict:
        loss += 0.5 * l2['body'] * get_dot_product(params, params)
    else:
        if 'base_params' in params:
            loss += 0.5 * l2 * get_dot_product(params['tangent_params'], params['tangent_params'])
        else:
            loss += 0.5 * l2 * get_dot_product(params, params)


    acc = jnp.mean(outputs.argmax(1) == labels.argmax(1))
    n_correct = jnp.sum(outputs.argmax(1) == labels.argmax(1))
    
    loss = loss/(labels.shape[0] * labels.shape[1])

    if has_bn and train:
        new_batch_stats = new_batch_stats['batch_stats']

    return loss, [new_batch_stats, acc, n_correct]

@functools.partial(jax.jit, static_argnames=('has_bn', 'train', 'update_ema', 'aug', 'use_base_params', 'do_krr', 'centering', 'max_batch_size', 'batch_size', 'alg_config'))
def do_training_steps(train_state, training_batch, key, n_steps = 100, l2 = 0., has_bn = False, train = True, update_ema = False, ema_decay = 0.995, aug = identity, training_batch2 = None, use_base_params = False, do_krr = False, centering = False, init_params = None, init_batch_params = None, batch_size = None, max_batch_size = None, inject_lr = None, alg_config = None):
    losses = jnp.zeros(1000)

    if inject_lr is not None:
        if not alg_config.naive_loss:
            train_state = train_state.replace(tx = optax.chain(
            optax.masked(optax.adam(learning_rate=inject_lr), {'base_params': False,  'tangent_params': get_tree_mask(model_depth = alg_config.model_depth, has_bn = has_bn, learn_final = alg_config.naive_loss)}),
            optax.masked(optax.adam(learning_rate=alg_config.pool_learning_rate), {'base_params': True,  'tangent_params': False})))
        else:
            train_state = train_state.replace(tx = optax.adam(learning_rate=inject_lr))
    
    
    def body_fn(i, val):
        train_state, losses, key = val
        
        if do_krr:
            aug_images = aug(key, training_batch['images'])
            batch_labels = training_batch['labels']

            if batch_size is None:
                grad_indices = None
            else:
                grad_indices = jax.random.choice(key, aug_images.shape[0], shape = [batch_size], replace = False)

            new_train_state, loss = do_training_step_krr(train_state, {'images': aug_images, 'labels': batch_labels}, l2 = l2, has_bn = has_bn, train = train, update_ema = update_ema, ema_decay = ema_decay, grad_indices = grad_indices, max_forward_batch_size = alg_config.max_forward_batch_size, use_x64 = alg_config.use_x64)
        else:
            if batch_size is None:
                aug_images = aug(key, training_batch['images'])
                batch_labels = training_batch['labels']
            else:
                key, aug_key, batch_key = jax.random.split(key, 3)
                batch_indices = jax.random.choice(batch_key, max_batch_size, shape = [batch_size], replace = False)
                aug_images = aug(aug_key, training_batch['images'][batch_indices])
                batch_labels = training_batch['labels'][batch_indices]

            new_train_state, loss = do_training_step(train_state, {'images': aug_images, 'labels': batch_labels}, l2 = l2, has_bn = has_bn, train = train, update_ema = update_ema, ema_decay = ema_decay, use_base_params = use_base_params, centering = centering, init_params = init_params, init_batch_params = init_batch_params)
        
        new_losses = losses.at[i].set(loss)
        key = jax.random.split(key)[0]
        
        return new_train_state, new_losses, key
    
    train_state, losses, _ = jax.lax.fori_loop(0, n_steps, body_fn, (train_state, losses, key))
    
    return train_state, losses



@functools.partial(jax.jit, static_argnames=('has_bn', 'train', 'update_ema', 'use_base_params', 'centering'))
def do_training_step(train_state, training_batch, l2 = 0., has_bn = False, train = True, update_ema = False, ema_decay = 0.995, use_base_params = False, centering = False, init_params = None, init_batch_params = None):
    images = training_batch['images']
    labels = training_batch['labels']
    
    if has_bn:
        batch_stats = train_state.batch_stats
    else:
        batch_stats = None
        
    (loss, (new_batch_stats, acc, _)), grad = jax.value_and_grad(get_training_loss_l2, argnums = 0, has_aux = True)(train_state.params, images, labels, train_state, l2 = l2, train = train, has_bn = has_bn, batch_stats = batch_stats, use_base_params = use_base_params, centering = centering, init_params = init_params, init_batch_params = init_batch_params)
    
        
    if has_bn:
        new_state = train_state.apply_gradients(grads = grad, batch_stats = new_batch_stats, train_it = train_state.train_it + 1)
    else:
        new_state = train_state.apply_gradients(grads = grad, train_it = train_state.train_it + 1)
    
    if update_ema:
        new_ema_hidden, new_ema_average = get_updated_ema(new_state.params, new_state.ema_hidden, ema_decay, new_state.train_it, order = 1)
        new_state = new_state.replace(ema_average = new_ema_average, ema_hidden = new_ema_hidden)
    
    return new_state, loss

@functools.partial(jax.jit, static_argnames=('has_bn', 'train', 'update_ema', 'use_base_params', 'max_forward_batch_size', 'use_x64'))
def do_training_step_krr(train_state, training_batch1, l2 = 0., has_bn = False, train = True, update_ema = False, ema_decay = 0.995, use_base_params = False, grad_indices = None, max_forward_batch_size = None, use_x64 = False):
    images1 = training_batch1['images']
    labels1 = training_batch1['labels']
    
    if has_bn:
        batch_stats = train_state.batch_stats
    else:
        batch_stats = None
        
    (loss, (new_batch_stats, acc, _)), grad = jax.value_and_grad(get_krr_loss_gd, argnums = 0, has_aux = True)(train_state.params, train_state, images1, labels1, None, None, l2 = l2, has_bn = has_bn, batch_stats = batch_stats, use_base_params = use_base_params, self_loss = True, grad_indices = grad_indices, max_forward_batch_size = max_forward_batch_size, use_x64 = use_x64)
    
        
    if has_bn:
        new_state = train_state.apply_gradients(grads = grad, batch_stats = new_batch_stats, train_it = train_state.train_it + 1)
    else:
        new_state = train_state.apply_gradients(grads = grad, train_it = train_state.train_it + 1)
    
    if update_ema:
        new_ema_hidden, new_ema_average = get_updated_ema(new_state.params, new_state.ema_hidden, ema_decay, new_state.train_it, order = 1)
        new_state = new_state.replace(ema_average = new_ema_average, ema_hidden = new_ema_hidden)

    
    return new_state, loss


def eval_on_test_set(train_state, test_loader, has_bn = False, use_ema = False, centering = False, init_params = None, init_batch_params = None):
    n_total = 0
    n_correct = 0
    
    params_to_use = train_state.params
    if use_ema:
        params_to_use = train_state.ema_average
    
    if has_bn:
        batch_stats = train_state.batch_stats
    else:
        batch_stats = None
    
    for images, labels in test_loader.as_numpy_iterator():
        _, (_, _, n) = get_training_loss_l2(params_to_use, images, labels, train_state, l2 = 0, train = False, has_bn = has_bn, batch_stats = batch_stats, centering = centering, init_params = init_params, init_batch_params = init_batch_params)
        n_correct += n
        n_total += labels.shape[0]
        
    return n_correct/n_total
            
            
class TrainStateWithBatchStats(train_state.TrainState):
    batch_stats: flax.core.FrozenDict
    train_it: int
    ema_hidden: Any = None
    ema_average: Any = None
    base_params: Any = None
    
class CoresetTrainState(train_state.TrainState):
    #A version of the train state that also returns the update when we apply gradients

    train_it: int
    ema_hidden: Any = None
    ema_average: Any = None

    def apply_gradients_get_updates(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(
            grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        ), updates
    
class ProtoHolder(nn.Module):
    x_proto_init: Any
    y_proto_init: Any
    temp_init: Any
    num_prototypes: int
    learn_label: bool = True
    use_flip: bool = False

    @nn.compact
    def __call__(self, ):
        x_proto = self.param('x_proto', lambda *_: self.x_proto_init)
        y_proto = self.param('y_proto', lambda *_: self.y_proto_init)
        log_temp = self.param('log_temp', lambda *_: self.temp_init)
        if not self.learn_label:
            y_proto = jax.lax.stop_gradient(y_proto)
        if self.use_flip:
            return jnp.concatenate([x_proto, jnp.flip(x_proto, axis = -2)], axis = 0), jnp.concatenate([y_proto, y_proto], axis = 0), log_temp

        return x_proto, y_proto, log_temp
    
def get_linear_forward(net_model_apply, has_bn = False, linearize = True):        
    if has_bn:
        def inner_fn(inner_params, images, batch_stats, **kwargs):
            return net_model_apply({'params': inner_params, 'batch_stats': batch_stats}, images, **kwargs)
        
        def linear_forward(variables_dict, images, use_base_params = False, add_primals = False, return_all = False, **kwargs):
            if use_base_params:
                return net_model_apply({'params': variables_dict['params']['base_params'], 'batch_stats': variables_dict['batch_stats']}, images, **kwargs)
            else:
                base_variables_dict = jax.lax.stop_gradient(variables_dict['params']['base_params'])

                if linearize:
                    primals, duals, aux = jax.jvp(utils.bind(inner_fn, ... , images, variables_dict['batch_stats'], **kwargs), (base_variables_dict,), (variables_dict['params']['tangent_params'],), has_aux = True)

                else:
                    primals, aux = inner_fn(base_variables_dict, images, variables_dict['batch_stats'], **kwargs)
                    dual_variable_dict = utils._add(base_variables_dict, variables_dict['params']['tangent_params'])
                    duals, _ = inner_fn(dual_variable_dict, images, variables_dict['batch_stats'], **kwargs)
                    duals = utils._sub(duals, primals)
                
                if return_all:
                    return primals, duals, aux

                if add_primals:
                    return utils._add(primals, duals), aux
                
                return duals, aux

    else:
        def inner_fn(inner_params, images, **kwargs):
            return net_model_apply({'params': inner_params}, images, **kwargs)
        
        def linear_forward(variables_dict, images, use_base_params = False, add_primals = False, return_all = False, **kwargs):
            if use_base_params:
                return net_model_apply({'params': variables_dict['params']['base_params']}, images, **kwargs)
            else:
                base_variables_dict = jax.lax.stop_gradient(variables_dict['params']['base_params'])
                
                if linearize:
                    primals, duals, aux = jax.jvp(utils.bind(inner_fn, ... , images, **kwargs), (base_variables_dict,), (variables_dict['params']['tangent_params'],), has_aux=True)

                else:
                    primals, aux = inner_fn(base_variables_dict, images, **kwargs)
                    dual_variable_dict = utils._add(base_variables_dict, variables_dict['params']['tangent_params'])
                    duals, _ = inner_fn(dual_variable_dict, images, **kwargs)
                    duals = utils._sub(duals, primals)
                
                if return_all:
                    return primals, duals, aux

                if add_primals:
                    return utils._add(primals, duals), aux
                
                return duals, aux
    
    return linear_forward

def get_training_loss_hvp(params, images, labels, net_train_state, v, l2 = 0, has_bn = False, batch_stats = None):
    def hvp(primals, tangents):
        return jax.jvp(jax.grad(utils.bind(get_training_loss_l2, ..., images, labels, net_train_state, l2 = l2, has_bn = has_bn, batch_stats = batch_stats, train = False), has_aux = True), [primals], [tangents], has_aux = True)[1]
    
    return hvp(params, v)

def get_training_loss_hvp_krr(params, coreset_images, coreset_labels, net_train_state, v, l2 = 0, has_bn = False, batch_stats = None, grad_indices = None, pre_s = None, alg_config = None):
    def hvp(primals, tangents):
        return jax.jvp(jax.grad(utils.bind(get_krr_loss_gd, ..., net_train_state, coreset_images, coreset_labels, None, None, l2 = l2, has_bn = has_bn, batch_stats = batch_stats, self_loss = True, grad_indices = grad_indices, pre_s = pre_s, max_forward_batch_size = alg_config.max_forward_batch_size, use_x64 = alg_config.use_x64), has_aux = True), [primals], [tangents], has_aux = True)[1]
    
    return hvp(params, v)


@jax.jit
def _bias_correction(moment, decay, count):
    """Perform bias correction. This becomes a no-op as count goes to infinity."""
    bias_correction = 1 - decay ** count
    return jax.tree_map(lambda t: t / bias_correction.astype(t.dtype), moment)


@jax.jit
def _update_moment(updates, moments, decay, order):
    """Compute the exponential moving average of the `order`-th moment."""
    return jax.tree_map(
        lambda g, t: (1 - decay) * (g ** order) + decay * t, updates, moments)

@jax.jit
def get_updated_ema(updates, moments, decay, count, order = 1):
    hidden = _update_moment(updates, moments, decay, order)
    average = _bias_correction(hidden, decay, count)
    
    return hidden, average


def multiply_by_scalar(x, s):
    return jax.tree_util.tree_map(lambda x: s * x, x)

def get_dot_product(a, b):
    return jnp.sum(sum_tree(utils._multiply(a, b)))

def sum_reduce(a, b):
    return jnp.sum(a) + jnp.sum(b)

def sum_tree(x):
    return jax.tree_util.tree_reduce(sum_reduce , x)

def init_proto(ds, num_prototypes_per_class, num_classes, class_subset=None, seed=0, scale_y=False, random_noise = False):
    window_size = num_prototypes_per_class
    reduce_func = lambda key, dataset: dataset.batch(window_size)
    ds = ds.shuffle(num_prototypes_per_class * num_classes * 10, seed=seed)
    ds = ds.group_by_window(key_func=lambda x, y: y, reduce_func=reduce_func, window_size=window_size)

    if class_subset is None:
        is_init = [0] * num_classes
    else:
        is_init = [1] * num_classes
        for cls in class_subset:
            is_init[cls] = 0

    x_proto = [None] * num_classes
    y_proto = [None] * num_classes
    for ele in ds.as_numpy_iterator():
        cls = ele[1][0]
        if is_init[cls] == 1:
            pass
        else:
            x_proto[cls] = ele[0]
            y_proto[cls] = ele[1]
            is_init[cls] = 1
        if sum(is_init) == num_classes:
            break
    x_proto = np.concatenate([x for x in x_proto if x is not None], axis=0)
    y_proto = np.concatenate([y for y in y_proto if y is not None], axis=0)
    y_proto = jax.nn.one_hot(y_proto, num_classes)

    if random_noise:
        np.random.seed(seed)
        x_proto = 0.3 * np.random.standard_normal(x_proto.shape)

    # center and scale y_proto
    y_proto = y_proto - 1 / num_classes
    if scale_y:
        y_scale = np.sqrt(num_classes / 10)
        y_proto = y_proto / y_scale
    return x_proto, y_proto