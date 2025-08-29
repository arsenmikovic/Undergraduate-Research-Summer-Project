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

def gradient_sensitivity_penalty(params, cppn, n_points=10):
    # Sample n_points random spatial positions in the image (x, y coordinates)
    x = y = jnp.linspace(-1, 1, 256)
    coords = jnp.stack(jnp.meshgrid(x, y, indexing='ij'), axis=-1)  # shape: (256, 256, 2)
    coords = coords.reshape(-1, 2)
    indices = jax.random.choice(jax.random.PRNGKey(0), coords.shape[0], shape=(n_points,), replace=False)
    sampled_coords = coords[indices]  # shape: (n_points, 2)

    def single_output(params, coord):
        inputs = {
            'x': coord[0],
            'y': coord[1],
            'd': jnp.sqrt(coord[0]**2 + coord[1]**2) * 1.4,
            'b': 1.0,
            'xabs': jnp.abs(coord[0]),
            'yabs': jnp.abs(coord[1]),
        }
        input_stack = jnp.stack([inputs[name] for name in cppn.cppn.inputs.split(",")])
        structured = cppn.param_reshaper.reshape_single(params)
        (h, s, v), _ = cppn.cppn.apply(structured, input_stack)
        r, g, b = hsv2rgb((h+1)%1, jnp.clip(s, 0, 1), jnp.clip(jnp.abs(v), 0, 1))
        rgb = jnp.stack([r, g, b])
        return rgb

    # Get gradient of RGB output w.r.t. params for each point
    def point_grad(coord):
        return jax.flatten_util.ravel_pytree(jax.grad(lambda p: jnp.sum(single_output(p, coord)))(params))[0]

    grads = jax.vmap(point_grad)(sampled_coords)
    grad_norms = jnp.sum(grads**2, axis=1)
    return jnp.mean(grad_norms)

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
    print('UPDATED')
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
    
    def loss_fn(params, target_img):
        img = cppn.generate_image(params, img_size=256)
        img_loss = jnp.mean((img - target_img)**2)

        grad_penalty = gradient_sensitivity_penalty(params, cppn)
        return img_loss + args.lambda_grad * grad_penalty

    # the dummy variable bellow _ is for jax.lax.scan
    @jax.jit
    def train_step(state, _):
        loss, grad = jax.value_and_grad(loss_fn)(state.params, target_img)
        grad = grad / (jnp.linalg.norm(grad) + 1e-8)  # normalize gradient
        state = state.apply_gradients(grads=grad)
        return state, loss

    #ADAM optimizer
    # option one is to add L2 regularization weight decay
    #tx = optax.adam(learning_rate=args.lr)
    scheduler = optax.cosine_decay_schedule(
        init_value=args.lr,
        decay_steps=args.n_iters,
        alpha=0.0  # final lr = alpha * init_value
        )
    tx = optax.chain(
        optax.add_decayed_weights(args.l2_coeff),
        optax.adam(learning_rate=scheduler)
    )
    state = TrainState.create(apply_fn=None, params=params, tx=tx)

    gen_img_fn = jax.jit(partial(cppn.generate_image, img_size=256))
    losses, imgs_train = [], [gen_img_fn(state.params)]
    pbar = tqdm(range(args.n_iters//100))
    for i_iter in pbar:
        state, loss = jax.lax.scan(train_step, state, None, length=100)
        # state, (loss, grad_norm) = jax.lax.scan(train_step, state, None, length=1)
        # print(loss, grad_norm)
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



