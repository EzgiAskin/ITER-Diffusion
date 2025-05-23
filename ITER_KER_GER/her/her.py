import os

import click
import numpy as np
import json
from mpi4py import MPI
from scipy.linalg import sqrtm

from baselines import logger
from baselines.common import set_global_seeds, tf_util
from baselines.common.mpi_moments import mpi_moments
import baselines.her.experiment.config as config
from baselines.her.rollout import RolloutWorker
from ipdb import set_trace
from tensorboardX import SummaryWriter
from baselines.her.ker_learning_method import SINGLE_SUC_RATE_THRESHOLD,IF_CLEAR_BUFFER
from baselines.her.ddpm_temporal import DDPM_Temporal
from baselines.her.util import (
    import_function, store_args, flatten_grads, transitions_in_episode_batch, convert_episode_to_batch_major)
from baselines.her.replay_buffer import ReplayBuffer


writer = SummaryWriter()



def mpi_average(value):
    if not isinstance(value, list):
        value = [value]
    if not any(value):
        value = [0.]
    return mpi_moments(np.array(value))[0]

def compute_success_rate(buffer, source='all'):
    """
    Compute the average success rate per episode for real or synthetic data
    """
    flags = buffer.synthetic_flags[:buffer.current_size]
    if source == 'real':
        available_idxs = np.where(~flags)[0]
    elif source == 'synthetic':
        available_idxs = np.where(flags)[0]
    else:
        available_idxs = np.arange(buffer.current_size)

    num_available = len(available_idxs)
    if num_available == 0:
        return float('nan'), float('nan')

    episodes = buffer.sample_episodes(num_available, source=source)
    rates = [ep['info_is_success'].mean() for ep in episodes]
    return np.mean(rates), np.std(rates)


def compute_feature_statistics(buffer, feature='o', source='all'):
    """
    Compute mean and covariance of a feature (e.g., states 'o' or actions 'u')
    across all timesteps and episodes.
    """
    flags = buffer.synthetic_flags[:buffer.current_size]
    if source == 'real':
        available_idxs = np.where(~flags)[0]
    elif source == 'synthetic':
        available_idxs = np.where(flags)[0]
    else:
        available_idxs = np.arange(buffer.current_size)

    if len(available_idxs) == 0:
        return None, None

    data = []
    for idx in available_idxs:
        ep = {key: buffer.buffers[key][idx] for key in buffer.buffers}
        feat = ep[feature]
        data.append(feat.reshape(-1, feat.shape[-1]))
    data = np.vstack(data)

    mean = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)
    return mean, cov


def kl_divergence_gaussian(mean1, cov1, mean2, cov2):
    """
    Compute the Kullback-Leibler (KL) divergence between two multivariate Gaussian distributions.

    Parameters:
    - mean1, cov1: Mean and covariance of the first distribution.
    - mean2, cov2: Mean and covariance of the second distribution.

    Returns:
    - KL divergence value (float).
    """
    dim = mean1.shape[0]
    inv_cov2 = np.linalg.inv(cov2 + np.eye(dim) * 1e-8)
    diff = mean2 - mean1
    term1 = np.trace(inv_cov2 @ cov1)
    term2 = diff.T @ inv_cov2 @ diff
    term3 = np.log((np.linalg.det(cov2) + 1e-8) / (np.linalg.det(cov1) + 1e-8))
    return 0.5 * (term1 + term2 - dim + term3)

def frechet_inception_distance(mean1, cov1, mean2, cov2):
    """
       Compute the Fréchet Inception Distance (FID) between two multivariate Gaussians.

       Parameters:
       - mean1, cov1: Mean and covariance of the first distribution.
       - mean2, cov2: Mean and covariance of the second distribution.

       Returns:
       - FID value (float).
    """
    diff = mean1 - mean2
    cov_sqrt, _ = sqrtm(cov1 @ cov2, disp=False)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real
    fid = diff.dot(diff) + np.trace(cov1 + cov2 - 2 * cov_sqrt)
    return np.real(fid)

def evaluate_buffer(buffer, features=('o', 'u', 'ag', 'g')):
    """
    Compute metrics comparing real and synthetic episodes:
      - success rates
      - KL & FID for each feature in `features`
    Returns a flat dict with keys like 'kl_o', 'fid_o', etc.
    """
    metrics = {}
    real_sr, real_sr_std = compute_success_rate(buffer, source='real')
    syn_sr, syn_sr_std   = compute_success_rate(buffer, source='synthetic')
    metrics['success_rate'] = {
        'real_mean': real_sr, 'real_std': real_sr_std,
        'synthetic_mean': syn_sr, 'synthetic_std': syn_sr_std
    }

    for feat in features:
        mu_r, cov_r = compute_feature_statistics(buffer, feature=feat, source='real')
        mu_s, cov_s = compute_feature_statistics(buffer, feature=feat, source='synthetic')

        if mu_r is None or mu_s is None:
            metrics[f'kl_{feat}']  = None
            metrics[f'fid_{feat}'] = None
        else:
            metrics[f'kl_{feat}']  = kl_divergence_gaussian(mu_r, cov_r, mu_s, cov_s)
            metrics[f'fid_{feat}'] = frechet_inception_distance(mu_r, cov_r, mu_s, cov_s)

    return metrics


def train(*, policy, rollout_worker, evaluator,
          n_epochs, n_test_rollouts, n_cycles, n_batches, policy_save_interval,
          save_path, demo_file, env_name,n_KER, ddpm, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()

    if save_path:
        latest_policy_path = os.path.join(save_path, 'policy_latest.pkl')
        best_policy_path = os.path.join(save_path, 'policy_best.pkl')
        periodic_policy_path = os.path.join(save_path, 'policy_{}.pkl')

    logger.info("Training...")
    best_success_rate = -1

    if policy.bc_loss == 1: policy.init_demo_buffer(demo_file) #initialize demo buffer if training with demonstrations

    # num_timesteps = n_epochs * n_cycles * rollout_length * number of rollout workers
    n_KER_number = n_KER
    first_time_enter = True
    test_suc_rate = 0
    single_suc_rate_threshold = SINGLE_SUC_RATE_THRESHOLD
    terminate_ker_now = False
    if_clear_buffer = False

    dims = policy.input_dims

    for epoch in range(n_epochs):
        rollout_worker.clear_history()
        for cycle in range(n_cycles):
            # generate episodes
            episodes = rollout_worker.generate_rollouts(terminate_ker=terminate_ker_now)
            # with KER
            # if (n_KER_number !=0) and terminate_ker_now==False:
            if (n_KER_number !=0):
                for episode in episodes:
                    policy.store_episode(episode)
            # without KER
            else:
                policy.store_episode(episodes)
                # HER/DDPG do not need clear buffer
                if_clear_buffer = False

            # Train only on last cycle of every epoch; after a certain epoch
            if cycle == n_cycles - 1 and epoch > 150:
                # Train DDPM on real episodes only
                real_eps = policy.buffer.sample_episodes(num_episodes=500, source='real')
                ddpm_losses = ddpm.train_ddpm(real_eps, batch_size=64, epochs=10,  return_losses=True)
                # Log ddpm loss
                logger.record_tabular('ddpm/loss_mean', np.mean(ddpm_losses))
                logger.record_tabular('ddpm/loss_std', np.std(ddpm_losses))
                # Log buffer stats
                buffer_stats = policy.buffer.get_buffer_stats()
                logger.record_tabular('buffer/episode_capacity', buffer_stats['episode_capacity'])
                logger.record_tabular('buffer/current_episodes', buffer_stats['current_episodes'])
                logger.record_tabular('buffer/transitions_stored', buffer_stats['transitions_stored'])
                logger.record_tabular('buffer/real_eps', buffer_stats['real_episodes'])
                logger.record_tabular('buffer/synth_eps', buffer_stats['synthetic_episodes'])

                if epoch % 20 == 0 and epoch > 200:
                    # Generate, tag and store synthetic episodes
                    ddpm.generate_and_store_synthetic_data(num_synthetic_episodes=1000)

                    # Log evaluation metrics for the ddpm
                    metrics = evaluate_buffer(policy.buffer)
                    sr = metrics['success_rate']
                    for feat in ('o', 'u', 'ag', 'g'):
                        kl = metrics[f'kl_{feat}']
                        fid = metrics[f'fid_{feat}']
                        if kl is not None:
                            logger.record_tabular(f'ddpm/kl_{feat}', kl)
                            logger.record_tabular(f'ddpm/fid_{feat}', fid)

                    logger.record_tabular('ddpm/real_sr_mean', sr['real_mean'])
                    logger.record_tabular('ddpm/real_sr_std', sr['real_std'])
                    logger.record_tabular('ddpm/synth_sr_mean', sr['synthetic_mean'])
                    logger.record_tabular('ddpm/synth_sr_std', sr['synthetic_std'])

            for _ in range(n_batches):
                policy.train()
            policy.update_target_net()
        policy.save(save_path)

        # test
        evaluator.clear_history()
        for _ in range(n_test_rollouts):
            evaluator.generate_rollouts()

        # record logs
        logger.record_tabular('epoch', epoch)
        for key, val in evaluator.logs('test'):
            logger.record_tabular(key, mpi_average(val))
            if key == "test/success_rate":
                test_suc_rate = val.copy()
        for key, val in rollout_worker.logs('train'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in policy.logs():
            logger.record_tabular(key, mpi_average(val))

        if rank == 0:
            logger.dump_tabular()

        # save the policy if it's better than the previous ones
        success_rate = mpi_average(evaluator.current_success_rate())
        writer.add_scalar(env_name+'_success_rate', success_rate, epoch)
        # if rank == 0 and success_rate >= best_success_rate and save_path:
        #     best_success_rate = success_rate
        #     logger.info('New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
        #     evaluator.save_policy(best_policy_path)
        #     evaluator.save_policy(latest_policy_path)
        # if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_path:
        #     policy_path = periodic_policy_path.format(epoch)
        #     logger.info('Saving periodic policy to {} ...'.format(policy_path))
        #     evaluator.save_policy(policy_path)

        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]
    writer.close()
    return policy


def learn(*, network, env, total_timesteps,
    seed=None,
    eval_env=None,
    replay_strategy='future',
    policy_save_interval=5,
    clip_return=True,
    demo_file=None,
    override_params=None,
    load_path=None,
    save_path=None,
    n_KER = 0,
    before_GER_minibatch_size = None,
    n_GER = 0,
    err_distance=0.05,
    ddpm_time_steps = 100,
    ddpm_tol = 0.05,
    **kwargs
):

    override_params = override_params or {}
    if MPI is not None:
        rank = MPI.COMM_WORLD.Get_rank()
        num_cpu = MPI.COMM_WORLD.Get_size()

    # Seed everything.
    rank_seed = seed + 1000000 * rank if seed is not None else None
    set_global_seeds(rank_seed)

    # Prepare params.
    params = config.DEFAULT_PARAMS
    if before_GER_minibatch_size is not None and n_GER is not None :
        params['batch_size'] = before_GER_minibatch_size * (n_GER+1)
    env_name = env.spec.id
    params['env_name'] = env_name
    params['replay_strategy'] = replay_strategy
    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    params.update(**override_params)  # makes it possible to override any parameter
    with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
         json.dump(params, f)
    params = config.prepare_params(params)
    params['rollout_batch_size'] = env.num_envs

    if demo_file is not None:
        params['bc_loss'] = 1
    params.update(kwargs)

    config.log_params(params, logger=logger)

    if num_cpu == 1:
        logger.warn()
        logger.warn('*** Warning ***')
        logger.warn(
            'You are running HER with just a single MPI worker. This will work, but the ' +
            'experiments that we report in Plappert et al. (2018, https://arxiv.org/abs/1802.09464) ' +
            'were obtained with --num_cpu 19. This makes a significant difference and if you ' +
            'are looking to reproduce those results, be aware of this. Please also refer to ' +
            'https://github.com/openai/baselines/issues/314 for further details.')
        logger.warn('****************')
        logger.warn()

    dims = config.configure_dims(params)
    policy = config.configure_ddpg(dims=dims, params=params, clip_return=clip_return,
                                    n_GER=n_GER,err_distance=err_distance,env_name=env_name)

    ddpm = DDPM_Temporal(policy.buffer, time_steps=ddpm_time_steps, tol=ddpm_tol)
    logger.info()
    logger.info("=== DDPM Configuration ===")
    logger.info(" time_steps: ", ddpm_time_steps)
    logger.info(" tolerance: ", ddpm_tol)
    logger.info("==========================")

    if load_path is not None:
        tf_util.load_variables(load_path)

    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': False,
        'T': params['T'],
    }

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'use_demo_states': False,
        'compute_Q': True,
        'T': params['T'],
    }

    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    eval_env = eval_env or env

    rollout_worker = RolloutWorker(env_name, env, policy, dims, logger, monitor=True,n_KER=n_KER, **rollout_params)
    evaluator = RolloutWorker(env_name,eval_env, policy, dims, logger, **eval_params)

    n_cycles = params['n_cycles']
    n_epochs = total_timesteps // n_cycles // rollout_worker.T // rollout_worker.rollout_batch_size

    return train(
        save_path=save_path, policy=policy, rollout_worker=rollout_worker,
        evaluator=evaluator, n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'],
        n_cycles=params['n_cycles'], n_batches=params['n_batches'],
        policy_save_interval=policy_save_interval, demo_file=demo_file,env_name=env_name, n_KER = n_KER, ddpm=ddpm)


@click.command()
@click.option('--env', type=str, default='FetchReach-v1', help='the name of the OpenAI Gym environment that you want to train on')
@click.option('--total_timesteps', type=int, default=int(5e5), help='the number of timesteps to run')
@click.option('--seed', type=int, default=0, help='the random seed used to seed both the environment and the training code')
@click.option('--policy_save_interval', type=int, default=5, help='the interval with which policy pickles are saved. If set to 0, only the best and latest policy will be pickled.')
@click.option('--replay_strategy', type=click.Choice(['future', 'none']), default='future', help='the HER replay strategy to be used. "future" uses HER, "none" disables HER.')
@click.option('--clip_return', type=int, default=1, help='whether or not returns should be clipped')
@click.option('--demo_file', type=str, default = 'PATH/TO/DEMO/DATA/FILE.npz', help='demo data file path')
def main(**kwargs):
    learn(**kwargs)


if __name__ == '__main__':
    main()
