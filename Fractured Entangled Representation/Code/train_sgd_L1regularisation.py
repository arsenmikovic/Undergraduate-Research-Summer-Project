import os
from functools import partial
import argparse

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax
from color import hsv2rgb

from cppn import CPPN, FlattenCPPNParameters
import util

parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0, help="the random seed")
group.add_argument("--save_dir", type=str, default=None, help="path to save results to")

group = parser.add_argument_group("model")
group.add_argument("--arch", type=str, default="", help="architecture")
group = parser.add_argument_group("data")
group.add_argument("--img_file", type=str, default=None, help="path of image file")

group = parser.add_argument_group("optimization")
group.add_argument("--n_iters", type=int, default=100000, help="number of iterations")
group.add_argument("--lr", type=float, default=3e-3, help="learning rate")
group.add_argument("--init_scale", type=str, default="default", help="initialization scale")
group.add_argument("--l2_coeff", type=float, default=0.0, help="L2 regularization coefficient")
group.add_argument("--lambda_grad", type=float, default=0.0, help="L2 regularization coefficient for gradient")

def weight_perturbation_penalty(params, cppn, img_size=256, sigma=1e-3, n_samples=1, rng_key=jax.random.PRNGKey(0)):
    """
    Penalizes the change in generated image due to small random perturbations in params.
    """
    img_orig = cppn.generate_image(params, img_size=img_size)

    def single_sample(rng):
        # Sample Gaussian noise for each param
        def add_noise(p):
            return p + sigma * jax.random.normal(rng, p.shape)
        
        perturbed_params = jax.tree_map(add_noise, params)
        img_perturbed = cppn.generate_image(perturbed_params, img_size=img_size)
        return jnp.mean((img_perturbed - img_orig) ** 2)

    rngs = jax.random.split(rng_key, n_samples)
    penalties = jax.vmap(single_sample)(rngs)
    return jnp.mean(penalties)


def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)  # set all "none" to None
    return args

def main(args):
    """
    Train a CPPN on a given image using SGD.
    Specify the architecture and img_file to train on.
    """
    print('UPDATED NEW')
    print(args)

    target_img = jnp.array(plt.imread(args.img_file)[:, :, :3])

    # initializes the CPPN and flatterns its parameters
    cppn = FlattenCPPNParameters(CPPN(args.arch, init_scale=args.init_scale))
    # cppn = FlattenCPPNParameters(CPPN(args.arch))

    rng = jax.random.PRNGKey(args.seed)
    params = cppn.init(rng)

    # def l2_regularization(params):
    #     leaves = jax.tree_util.tree_leaves(cppn.param_reshaper.reshape_single(params)['params'])
    #     return sum([jnp.sum(jnp.square(p)) for p in leaves if p.ndim >= 2])

    # def loss_fn(params, target_img):
    #     img = cppn.generate_image(params, img_size=256)
    #     img_loss = jnp.mean((img - target_img)**2)
    #     l2_loss = args.l2_coeff * l2_regularization(params)
    #     return img_loss + l2_loss

    # def loss_fn(params, target_img):
    #     img = cppn.generate_image(params, img_size=256)
    #     return jnp.mean((img - target_img)**2)

    def loss_fn(params, target_img, rng_key):
        img = cppn.generate_image(params, img_size=256)
        img_loss = jnp.mean((img - target_img)**2)

        # functional / weight perturbation regularization
        perturb_penalty = weight_perturbation_penalty(params, cppn, sigma=1e-3, n_samples=1, rng_key=rng_key)

        return img_loss + args.lambda_grad * perturb_penalty

    # the dummy variable bellow _ is for jax.lax.scan
    @jax.jit
    def train_step(state, rng):
        loss, grad = jax.value_and_grad(loss_fn)(state.params, target_img, rng)
        grad = grad / (jnp.linalg.norm(grad) + 1e-8)  # normalize gradient
        state = state.apply_gradients(grads=grad)
        return state, loss

    #ADAM optimizer
    # option one is to add L2 regularization weight decay
    #tx = optax.adam(learning_rate=args.lr)
    tx = optax.chain(optax.add_decayed_weights(args.l2_coeff), optax.adam(learning_rate=args.lr))
    state = TrainState.create(apply_fn=None, params=params, tx=tx)

    gen_img_fn = jax.jit(partial(cppn.generate_image, img_size=256))
    losses, imgs_train = [], [gen_img_fn(state.params)]
    pbar = tqdm(range(args.n_iters//100))
    rng = jax.random.PRNGKey(args.seed)
    for i_iter in pbar:
        rng, step_rng = jax.random.split(rng)
        num_steps = 100
        rng, *step_rngs = jax.random.split(rng, num_steps + 1)
        step_rngs = jnp.stack(step_rngs)  # shape (100, 2)

        state, loss = jax.lax.scan(train_step, state, step_rngs)
        losses.append(loss)
        pbar.set_postfix(loss=loss.mean().item())
        if i_iter < 100:
            img = gen_img_fn(state.params)
            imgs_train.append(img)

    losses = np.array(jnp.concatenate(losses))
    imgs_train = np.array(jnp.stack(imgs_train))
    params = state.params
    img = gen_img_fn(params)

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        util.save_pkl(args.save_dir, "args", args)
        util.save_pkl(args.save_dir, "arch", args.arch)
        util.save_pkl(args.save_dir, "params", params)
        plt.imsave(f"{args.save_dir}/img.png", np.array(img))

        util.save_pkl(args.save_dir, "losses", losses)
        # util.save_pkl(args.save_dir, "imgs_train", imgs_train)

if __name__ == '__main__':
    main(parse_args())



