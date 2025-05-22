import threading
import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_shapes, size_in_transitions, T, sample_transitions):
        """Creates a replay buffer with tagging for real vs synthetic data."""
        self.buffer_shapes = buffer_shapes
        self.size = size_in_transitions // T  # capacity in episodes
        self.T = T
        self.sample_transitions = sample_transitions

        # Each buffer key -> array[episode_idx, time, dim]
        self.buffers = {key: np.empty([self.size, *shape])
                        for key, shape in buffer_shapes.items()}
        # synthetic flag per episode
        self.synthetic_flags = np.zeros(self.size, dtype=bool)

        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0

        self.lock = threading.Lock()

    @property
    def full(self):
        with self.lock:
            return self.current_size == self.size

    def get_buffer_stats(self):
        """
        Return detailed buffer stats including real vs synthetic counts.
        """
        with self.lock:
            total_eps = self.current_size
            synthetic_eps = int(self.synthetic_flags[:total_eps].sum())
            real_eps = total_eps - synthetic_eps
            return {
                'episode_capacity': self.size,
                'current_episodes': total_eps,
                'transitions_stored': self.n_transitions_stored,
                'current_transitions': total_eps * self.T,
                'real_episodes': real_eps,
                'synthetic_episodes': synthetic_eps,
            }

    def sample(self, batch_size, env_name=None, n_GER=0, err_distance=0.05):
        """Samples transitions """
        with self.lock:
            assert self.current_size > 0
            # buffer full episodes
            buffers = {k: v[:self.current_size] for k, v in self.buffers.items()}

        buffers['o_2'] = buffers['o'][:, 1:, :]
        buffers['ag_2'] = buffers['ag'][:, 1:, :]
        transitions = self.sample_transitions(buffers, batch_size,
                                              env_name=env_name,
                                              n_GER=n_GER,
                                              err_distance=err_distance)
        # validation
        for key in (['r', 'o_2', 'ag_2'] + list(self.buffers.keys())):
            assert key in transitions, f"key {key} missing from transitions"

        return transitions

    def store_episode(self, episode_batch, synthetic=False):
        """Store episodes; tag as synthetic or real"""
        # episode_batch: dict of arrays [T or T+1]
        batch_size = next(iter(episode_batch.values())).shape[0]
        with self.lock:
            idxs = self._get_storage_idx(batch_size)

            for key in self.buffers.keys():
                self.buffers[key][idxs] = episode_batch[key]
            # tag
            self.synthetic_flags[idxs] = synthetic
            self.n_transitions_stored += batch_size * self.T

    def sample_episodes(self, num_episodes, source='all'):
        """Sample full episodes; can filter by source: 'all', 'real', or 'synthetic'."""
        with self.lock:
            if self.current_size == 0:
                raise ValueError("Buffer is empty")
            flags = self.synthetic_flags[:self.current_size]
            if source == 'real':
                candidates = np.where(~flags)[0]
            elif source == 'synthetic':
                candidates = np.where(flags)[0]
            else:
                candidates = np.arange(self.current_size)

            if num_episodes > len(candidates):
                raise ValueError(f"Requested {num_episodes} episodes, but only {len(candidates)} available for source={source}.")

            indices = np.random.choice(candidates, size=num_episodes, replace=False)
            episodes = []
            for idx in indices:
                ep = {key: self.buffers[key][idx].copy() for key in self.buffers.keys()}
                episodes.append(ep)
        return episodes

    def clear_buffer(self):
        with self.lock:
            self.current_size = 0
            self.n_transitions_stored = 0
            for key, shape in self.buffer_shapes.items():
                self.buffers[key] = np.empty([self.size, *shape])
            self.synthetic_flags[:] = False

    def _get_storage_idx(self, inc=1):
        assert inc <= self.size, "Batch too large!"
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx = np.concatenate([
                np.arange(self.current_size, self.size),
                np.random.randint(0, self.current_size, overflow)
            ])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        return idx if inc > 1 else idx[0]

    def sample_successful_contexts(self, num):
        """
        Return up to `num` (s0, g0) pairs drawn *only* from real episodes
        that had at least one successful timestep.
        """
        with self.lock:
            N = self.current_size
            # real-episode mask
            real_mask = ~self.synthetic_flags[:N]  # shape [N]
            # for each real episode, check if any success flag is True
            #    info_is_success has shape [N, T, 1], so we squeeze and any over axis=1
            infos = self.buffers['info_is_success'][:N, :, 0]  # [N, T]
            success_mask = infos.any(axis=1)  # [N]
            # combine to get indices of real & successful episodes
            good_idxs = np.where(real_mask & success_mask)[0]
            if len(good_idxs) == 0:
                raise ValueError("No successful real episodes available in buffer.")
            # sample (with replacement if not enough)
            replace = len(good_idxs) < num
            chosen = np.random.choice(good_idxs, size=num, replace=replace)

            contexts = []
            for idx in chosen:
                s0 = self.buffers['o'][idx, 0]  # initial state
                g0 = self.buffers['g'][idx, 0]  # initial goal
                contexts.append((s0, g0))
            return contexts


    def get_initial_values(self):
        """Return a random episode's initial (state, action). Todo Deprecated, Remove later"""
        with self.lock:
            if self.current_size == 0:
                raise ValueError("Buffer is empty")
            idx = np.random.randint(0, self.current_size)
            s0 = self.buffers['o'][idx, 0]
            u0 = self.buffers['u'][idx, 0]
        return s0, u0