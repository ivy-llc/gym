# global
import ivy_gym
import argparse
import numpy as np
from ivy_demo_utils.framework_utils import choose_random_framework, get_framework_from_str


def loss_fn(env, initial_state, logits_in, f):
    env.set_state(initial_state)
    score = f.array([0.])
    for logs_ in f.unstack(logits_in, 0):
        ac = f.tanh(logs_)
        rew = env.step(ac)[1]
        score = score + rew
    return -score[0]


def train_step(compiled_loss_fn, initial_state, logits, lr, f):
    loss, grads = f.execute_with_gradients(lambda lgts: compiled_loss_fn(initial_state, lgts[0]), [logits])
    logits = f.gradient_descent_update([logits], grads, lr)[0]
    return -f.reshape(loss, (1,)), logits


def main(env_str, steps=100, iters=10000, lr=0.1, seed=0, log_freq=100, vis_freq=1000, visualize=True, f=None):

    # config
    f = choose_random_framework(excluded=['numpy']) if f is None else f
    f.seed(seed)
    env = getattr(ivy_gym, env_str)(f=f)
    env.reset()
    starting_state = env.get_state()

    # trajectory parameters
    ac_dim = env.action_space.shape[0]
    logits = f.variable(f.random_uniform(-2, 2, (steps, ac_dim)))

    # compile loss function
    compiled_loss_fn = f.compile_fn(lambda initial_state, lgts: loss_fn(env, initial_state, lgts, f),
                                    example_inputs=[starting_state, logits])

    # Train
    scores = []
    for iteration in range(iters):

        if iteration % vis_freq == 0 and visualize:
            env.set_state(starting_state)
            env.render()
            for logs in f.unstack(logits, axis=0):
                ac = f.tanh(logs)
                env.step(ac)
                env.render()

        env.set_state(starting_state)
        if iteration == 0:
            print('\nCompiling loss function for {} environment steps... This may take a while...\n'.format(steps))
        score, logits = train_step(compiled_loss_fn, starting_state, logits, lr, f)
        if iteration == 0:
            print('\nLoss function compiled!\n')
        print('iteration {} score {}'.format(iteration, f.to_numpy(score).item()))
        scores.append(f.to_numpy(score)[0])

        if len(scores) == log_freq:
            print('\nIterations: {} Mean Score: {}\n'.format(iteration + 1, np.mean(scores)))
            scores.clear()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_visuals', action='store_true',
                        help='whether to run the demo without rendering images.')
    parser.add_argument('--env', default='CartPole',
                        choices=['CartPole', 'Pendulum', 'MountainCar', 'Reacher', 'Swimmer'])
    parser.add_argument('--framework', type=str, default=None,
                        help='which framework to use. Chooses a random framework if unspecified.')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--iters', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--vis_freq', type=int, default=1000)
    parsed_args = parser.parse_args()
    framework = get_framework_from_str(parsed_args.framework)
    if parsed_args.framework == 'numpy':
        raise Exception('Invalid framework selection. Numpy does not support auto-differentiation.\n'
                        'This demo involves gradient-based optimization, and so auto-diff is required.\n'
                        'Please choose a different backend framework.')
    print('\nTraining for {} iterations.\n'.format(parsed_args.iters))
    main(parsed_args.env, parsed_args.steps, parsed_args.iters, parsed_args.lr, parsed_args.seed,
         parsed_args.log_freq, parsed_args.vis_freq, not parsed_args.no_visuals, framework)
