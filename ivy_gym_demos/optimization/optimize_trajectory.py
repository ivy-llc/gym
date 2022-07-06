# global
import ivy
import ivy_gym
import argparse
import numpy as np


def loss_fn(env, initial_state, logits_in):
    env.set_state(initial_state)
    score = ivy.array([0.])
    for logs_ in ivy.unstack(logits_in, axis=0):
        ac = ivy.tanh(logs_)
        rew = env.step(ac)[1]
        score = score + rew
    return ivy.to_native(-score[0])


# def train_step(compiled_loss_fn, optimizer, initial_state, logits):
def train_step(env, optimizer, initial_state, logits):
    initial_state = ivy.to_native(initial_state, nested=True)
    # loss, grads = ivy.execute_with_gradients(lambda lgts: compiled_loss_fn(initial_state, lgts['l']),
                                             # ivy.Container({'l': logits}))
    loss, grads = ivy.execute_with_gradients(loss_fn(env, initial_state, logits))
    logits = optimizer.step(ivy.Container({'l': logits}), grads)['l']
    return -ivy.reshape(loss, (1,)), logits


def main(env_str, steps=100, iters=10000, lr=0.1, seed=0, log_freq=100, vis_freq=1000, visualize=True, f=None, fw=None):

    # config
    fw = ivy.choose_random_backend(excluded=['numpy']) if fw is None else fw
    ivy.set_backend(fw)
    f = ivy.get_backend(fw) if f is None else f
    ivy.seed(seed)
    env = getattr(ivy_gym, env_str)()
    env.reset()
    starting_state = env.get_state()

    # trajectory parameters
    ac_dim = env.action_space.shape[0]
    logits = ivy.variable(ivy.random_uniform(-2, 2, shape=(steps, ac_dim)))

    # compile loss function
    # compiled_loss_fn = ivy.compile(lambda initial_state, lgts: loss_fn(env, initial_state, lgts),
    #                                False, example_inputs=[ivy.to_native(starting_state, nested=True), ivy.to_native(logits)])

    # optimizer
    optimizer = ivy.Adam(lr=lr)

    # train
    scores = []
    for iteration in range(iters):

        if iteration % vis_freq == 0 and visualize:
            env.set_state(starting_state)
            env.render()
            for logs in ivy.unstack(logits, axis=0):
                ac = ivy.tanh(logs)
                env.step(ac)
                env.render()

        env.set_state(starting_state)
        if iteration == 0:
            print('\nCompiling loss function for {} environment steps... This may take a while...\n'.format(steps))
        # score, logits = train_step(compiled_loss_fn, optimizer, starting_state, logits)
        score, logits = train_step(env, optimizer, starting_state, logits)
        if iteration == 0:
            print('\nLoss function compiled!\n')
        print('iteration {} score {}'.format(iteration, ivy.to_numpy(score).item()))
        scores.append(ivy.to_numpy(score)[0])

        if len(scores) == log_freq:
            print('\nIterations: {} Mean Score: {}\n'.format(iteration + 1, np.mean(scores)))
            scores.clear()
    ivy.unset_backend()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_visuals', action='store_true',
                        help='whether to run the demo without rendering images.')
    parser.add_argument('--env', default='CartPole',
                        choices=['CartPole', 'Pendulum', 'MountainCar', 'Reacher', 'Swimmer'])
    parser.add_argument('--backend', type=str, default=None,
                        help='which framework to use. Chooses a random framework if unspecified.')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--iters', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--vis_freq', type=int, default=1000)
    parsed_args = parser.parse_args()
    fw = parsed_args.backend
    if fw is None:
        fw = ivy.choose_random_backend(excluded=['numpy'])
    if fw == 'numpy':
        raise Exception('Invalid framework selection. Numpy does not support auto-differentiation.\n'
                        'This demo involves gradient-based optimization, and so auto-diff is required.\n'
                        'Please choose a different backend framework.')
    f = ivy.get_backend(fw)
    print('\nTraining for {} iterations.\n'.format(parsed_args.iters))
    main(parsed_args.env, parsed_args.steps, parsed_args.iters, parsed_args.lr, parsed_args.seed,
         parsed_args.log_freq, parsed_args.vis_freq, not parsed_args.no_visuals, f, fw)
