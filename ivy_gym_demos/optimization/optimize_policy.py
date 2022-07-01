# global
import ivy
import ivy_gym
import argparse
import numpy as np


class Policy(ivy.Module):

    def __init__(self, in_size, out_size, hidden_size=64):
        self._linear0 = ivy.Linear(in_size, hidden_size)
        self._linear1 = ivy.Linear(hidden_size, hidden_size)
        self._linear2 = ivy.Linear(hidden_size, out_size)
        ivy.Module.__init__(self, device='cpu')

    def _forward(self, x):
        x = ivy.expand_dims(x, axis=0)
        x = ivy.tanh(self._linear0(x, v=self.v.linear0))
        x = ivy.tanh(self._linear1(x, v=self.v.linear1))
        return ivy.tanh(self._linear2(x, v=self.v.linear2))[0]


def loss_fn(env, initial_state, policy, v, steps):
    obs = env.set_state(initial_state)
    score = ivy.array([0.])
    for step in range(steps):
        ac = policy(obs, v=v)
        obs, rew, _, _ = env.step(ac)
        score = score + rew
    return ivy.to_native(-score[0])


def train_step(compiled_loss_fn, optimizer, initial_state, policy, f):
    initial_state = ivy.to_native(initial_state, nested=True)
    loss, grads = ivy.execute_with_gradients(lambda pol_vs: compiled_loss_fn(initial_state, pol_vs), policy.v)
    policy.v = optimizer.step(policy.v, grads)
    return -ivy.reshape(loss, (1,))


def main(env_str, steps=100, iters=10000, lr=0.001, seed=0, log_freq=100, vis_freq=1000, visualize=True, f=None, fw=None):

    # config
    fw = ivy.choose_random_backend() if fw is None else fw
    ivy.set_backend(fw)
    f = ivy.get_backend(fw)
    ivy.seed(seed)
    env = getattr(ivy_gym, env_str)()
    starting_obs = env.reset()

    # policy
    in_size = starting_obs.shape[0]
    ac_dim = env.action_space.shape[0]
    policy = Policy(in_size, ac_dim)

    # compile loss function
    compiled_loss_fn = ivy.compile(lambda initial_state, pol_vs:
                                   loss_fn(env, initial_state, policy, pol_vs, steps),
                                   False, example_inputs=[ivy.to_native(env.get_state(), nested=True), ivy.to_native(policy.v, nested=True)])

    # optimizer
    optimizer = ivy.Adam(lr=lr)

    # train
    scores = []
    for iteration in range(iters):

        if iteration % vis_freq == 0 and visualize:
            obs = env.reset()
            env.render()
            for _ in range(steps):
                ac = policy(obs)
                obs, _, _, _ = env.step(ac)
                env.render()

        env.reset()
        if iteration == 0:
            print('\nCompiling loss function for {} environment steps... This may take a while...\n'.format(steps))
        score = train_step(compiled_loss_fn, optimizer, env.get_state(), policy, f)
        if iteration == 0:
            print('\nLoss function compiled!\n')
        print('iteration {} score {}'.format(iteration, ivy.to_numpy(score).item()))
        scores.append(ivy.to_numpy(score)[0])

        if len(scores) == log_freq:
            print('\nIterations: {} Mean Score: {}\n'.format(iteration + 1, np.mean(scores)))
            scores.clear()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_visuals', action='store_true',
                        help='whether to run the demo without rendering images.')
    parser.add_argument('--env', default='CartPole',
                        choices=['CartPole', 'Pendulum', 'MountainCar', 'Reacher', 'Swimmer'])
    parser.add_argument('--backend', type=str, default=None,
                        choices=['jax', 'tensorflow', 'torch', 'mxnet', 'numpy'])
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--iters', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.001)
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
