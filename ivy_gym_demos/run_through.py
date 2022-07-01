# global
import ivy
import argparse
import ivy_gym


def main(env_str=None, visualize=True, f=None, fw=None):

    # Framework Setup #
    # ----------------#

    fw = ivy.choose_random_backend() if fw is None else fw
    ivy.set_backend(fw)
    f = ivy.get_backend(fw) if f is None else f

    # get environment
    env = getattr(ivy_gym, env_str)()

    # run environment steps
    env.reset()
    ac_dim = env.action_space.shape[0]
    for _ in range(250):
        ac = ivy.random_uniform(-1, 1, shape=(ac_dim,))
        env.step(ac)
        if visualize:
            env.render()
    env.close()
    ivy.unset_backend()

    # message
    print('End of Run Through Demo!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_visuals', action='store_true',
                        help='whether to run the demo without rendering images.')
    parser.add_argument('--env', default='CartPole',
                        choices=['CartPole', 'Pendulum', 'MountainCar', 'Reacher', 'Swimmer'])
    parser.add_argument('--backend', type=str, default=None,
                        help='which backend to use. Chooses a random backend if unspecified.')
    parsed_args = parser.parse_args()
    fw = parsed_args.backend
    f = None if fw is None else ivy.get_backend(fw)
    main(parsed_args.env, not parsed_args.no_visuals, f, fw)
