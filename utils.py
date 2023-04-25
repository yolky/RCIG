import functools
import jax
import operator
import numpy as np
import jax.numpy as jnp

class bind(functools.partial):
    """
    An improved version of partial which accepts Ellipsis (...) as a placeholder
    """
    def __call__(self, *args, **keywords):
        keywords = {**self.keywords, **keywords}
        iargs = iter(args)
        args = (next(iargs) if arg is ... else arg for arg in self.args)
        return self.func(*args, *iargs, **keywords)
    

def _sub(x, y):
    return jax.tree_util.tree_map(operator.sub, x, y)
    
def _add(x, y):
    return jax.tree_util.tree_map(operator.add, x, y)

def _multiply(x, y):
    return jax.tree_util.tree_map(operator.mul, x, y)

def _divide(x, y):
    return jax.tree_util.tree_map(operator.truediv, x, y)

def _one_like(x):
    return jax.tree_util.tree_map(lambda a: jnp.ones_like(a), x)

def get_class_indices(train_labels, samples_per_class, seed = 0, n_classes = 10):    
    np.random.seed(seed)
    combined_indices = []

    for c in range(n_classes):
        class_indices = np.where(train_labels.numpy() == c)[0]
        combined_indices.extend(class_indices[np.random.choice(len(class_indices), samples_per_class, replace = False)])

    return combined_indices


def _zero_like(x):
    return jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), x)


def multiply_by_scalar(x, s):
    return jax.tree_util.tree_map(lambda x: s * x, x)