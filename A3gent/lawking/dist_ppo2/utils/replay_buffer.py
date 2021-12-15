from lawking.dist_ppo2.utils.utils import *
from itertools import chain

class ReplayBuffer(object):

    def __init__(self, max_size=1000000):

        self.max_size = max_size
        self.paths = []
        self.path_list = []
        self.paths_len = 0
        self.obs = None
        self.acs = None
        self.concatenated_rews = None
        self.unconcatenated_rews = None
        self.next_obs = None
        self.terminals = None

    def add_rollouts(self, paths, noised=False):

        # add new rollouts into our list of rollouts
        # for path in paths:
        #     self.paths.append(path)

        # if len(self.path_list) == 2:
        #     self.path_list = self.path_list[1:]
        # self.path_list.append(paths)
        # self.paths = list(chain.from_iterable(self.path_list))

        # convert new rollouts into their component arrays, and append them onto our arrays
        observations, actions, next_observations, terminals, concatenated_rews, unconcatenated_rews = convert_listofrollouts(paths)
        del paths

        if noised:
            observations = add_noise(observations)
            next_observations = add_noise(next_observations)

        if self.obs is None:
            self.obs = observations[-self.max_size:]
            self.acs = actions[-self.max_size:]
            self.next_obs = next_observations[-self.max_size:]
            self.terminals = terminals[-self.max_size:]
            self.concatenated_rews = concatenated_rews[-self.max_size:]
            self.unconcatenated_rews = unconcatenated_rews[-self.max_size:]
        else:
            # print("=====", len(self.obs), len(observations), self.max_size)
            if len(self.obs) + len(observations) > self.max_size:
                self.obs = self.obs[-(self.max_size-len(observations)):]
                self.acs = self.acs[-(self.max_size-len(actions)):]
                self.next_obs = self.next_obs[-(self.max_size-len(next_observations)):]
                self.terminals = self.terminals[-(self.max_size-len(terminals)):]
                self.concatenated_rews = self.concatenated_rews[-(self.max_size-len(concatenated_rews)):]

                self.obs = np.concatenate([self.obs, observations])
                self.acs = np.concatenate([self.acs, actions])
                self.next_obs = np.concatenate([self.next_obs, next_observations])
                self.terminals = np.concatenate([self.terminals, terminals])
                self.concatenated_rews = np.concatenate([self.concatenated_rews, concatenated_rews])
                
                # print(self.obs.shape, self.max_size, len(concatenated_rews), self.max_size-len(concatenated_rews))
                # self.obs = np.concatenate([self.obs, observations])[-self.max_size:]
                # self.acs = np.concatenate([self.acs, actions])[-self.max_size:]
                # self.next_obs = np.concatenate(
                #     [self.next_obs, next_observations]
                # )[-self.max_size:]
                # self.terminals = np.concatenate(
                #     [self.terminals, terminals]
                # )[-self.max_size:]
                # self.concatenated_rews = np.concatenate(
                #     [self.concatenated_rews, concatenated_rews]
                # )[-self.max_size:]

                # self.obs = observations
                # self.acs = actions
                # self.next_obs = next_observations
                # self.terminals = terminals
                # self.concatenated_rews = concatenated_rews
            else:
                self.obs = np.concatenate([self.obs, observations])[-self.max_size:]
                self.acs = np.concatenate([self.acs, actions])[-self.max_size:]
                self.next_obs = np.concatenate(
                    [self.next_obs, next_observations]
                )[-self.max_size:]
                self.terminals = np.concatenate(
                    [self.terminals, terminals]
                )[-self.max_size:]
                self.concatenated_rews = np.concatenate(
                    [self.concatenated_rews, concatenated_rews]
                )[-self.max_size:]
            if isinstance(unconcatenated_rews, list):
                self.unconcatenated_rews += unconcatenated_rews  # TODO keep only latest max_size around
            else:
                self.unconcatenated_rews.append(unconcatenated_rews)  # TODO keep only latest max_size around

    ########################################
    ########################################

    # def sample_random_rollouts(self, num_rollouts):
    #     rand_indices = np.random.permutation(len(self.paths))[:num_rollouts]
    #     return self.paths[rand_indices]

    # def sample_recent_rollouts(self, num_rollouts=1):
    #     return self.paths[-num_rollouts:]

    ########################################
    ########################################

    def sample_random_data(self, batch_size):

        assert self.obs.shape[0] == self.acs.shape[0] == self.concatenated_rews.shape[0] == self.next_obs.shape[0] == self.terminals.shape[0]
        rand_indices = np.random.permutation(self.obs.shape[0])[:batch_size]
        return self.obs[rand_indices], self.acs[rand_indices], self.concatenated_rews[rand_indices], self.next_obs[rand_indices], self.terminals[rand_indices]

    # def sample_recent_data(self, batch_size=1, concat_rew=True):

    #     if concat_rew:
    #         return self.obs[-batch_size:], self.acs[-batch_size:], self.concatenated_rews[-batch_size:], self.next_obs[-batch_size:], self.terminals[-batch_size:]
    #     else:
    #         num_recent_rollouts_to_return = 0
    #         num_datapoints_so_far = 0
    #         index = -1
    #         while num_datapoints_so_far < batch_size:
    #             recent_rollout = self.paths[index]
    #             index -=1
    #             num_recent_rollouts_to_return +=1
    #             num_datapoints_so_far += get_pathlength(recent_rollout)
    #         rollouts_to_return = self.paths[-num_recent_rollouts_to_return:]
    #         observations, actions, next_observations, terminals, concatenated_rews, unconcatenated_rews = convert_listofrollouts(rollouts_to_return)
    #         return observations, actions, unconcatenated_rews, next_observations, terminals