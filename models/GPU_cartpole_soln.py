import gym
import torch
from torch.distributions import MultivariateNormal, Categorical
import matplotlib.pyplot as plt
import torch.nn.functional as F
from numpy import floor
import torch.multiprocessing as mp
from copy import deepcopy
import time
import torch.utils.data
from collections import OrderedDict

import os
import torch.distributed as dist

class POLICY(torch.nn.Module):
    """
    POLICY CLASS: CONTAINS ALL USABLE FUNCTIONAL FORMS FOR THE AGENTS POLICY, ALONG WITH
    THE CORROSPONDING HELPER FUNCTIONS, INITIALIZED FOR DISCRETE AND CONTINOUS ACTION
    SPACES.
    """
    def __init__(self, state_size, action_size, actions, hidden_layer = 32, epsilon = 0.02):
        super(POLICY, self).__init__()
        # initializations
        self.state_size = state_size
        self.action_size = action_size
        # 1 hidden relu layer
        self.linear1 = torch.nn.Linear(state_size, hidden_layer)
        # 2 hidden relu layer
        self.linear2 = torch.nn.Linear(hidden_layer, hidden_layer)
        # 2 hidden relu layer
        self.linear3 = torch.nn.Linear(hidden_layer, actions)
        # 3 output through softmax
        self.output = torch.nn.Softmax(dim=0)
        self.outputstacked = torch.nn.Softmax(dim=1)
        self.dist = lambda prob: Categorical(prob)

    def sample_action(self, state):
        # First Hidden Layer
        output = self.linear1(torch.FloatTensor(state))
        output = F.relu(output)
        # seconf Hidden Layer
        output = self.linear2(output)
        output = F.relu(output)
        # third Hidden Layer
        output = self.linear3(output)
        # output = F.relu(output)
        # outputs a parameterization
        output = self.output(output)
        # parameterized distribution
        distrib = self.dist(output)
        # return action
        return distrib.sample()

    def logprob_action(self, state, action):
        # Nueral net with two hidden relu layers
        output = self.linear1(state)
        output = F.relu(output)
        # seconf Hidden Layer
        output = self.linear2(output)
        output = torch.relu(output)
        # third Hidden Layer
        output = self.linear3(output)
        # output = F.relu(output)
        # outputs a parameterization
        output = self.output(output)
        # parameterized distribution
        distrib = self.dist(output)
        # return log probability of an action
        return distrib.log_prob(action)

    def forward(self, state, action):
        # Nueral net with two hidden relu layers
        output = self.linear1(state)
        output = F.relu(output)
        # seconf Hidden Layer
        output = self.linear2(output)
        output = torch.relu(output)
        # third Hidden Layer
        output = self.linear3(output)
        # output = F.relu(output)
        # outputs a parameterization
        output = self.outputstacked(output)
        # return log probability of an action
        return torch.log(torch.gather(output, 1, action))

class PG_LOSS(torch.nn.Module):
    """ POLICY GRADIENTS LOSS FUNCTION """
    def __init__(self, trajectory_length, simulations):
        """ INITIALIZATIONS """
        super(PG_LOSS, self).__init__()
        self.simulations = simulations
        self.trajectory_length = trajectory_length

    def forward(self, policy, state_tensor, action_tensor, reward_tensor, \
                        cumulative_rollout):

        """ CONVERT FORMAT """
        flat_states = torch.flatten(state_tensor, start_dim=0,end_dim=1)
        flat_actions = torch.flatten(action_tensor, start_dim=0,end_dim=1)
        flat_cumsum = torch.flatten(cumulative_rollout, start_dim=0,end_dim=1)

        """ CALCULATE LIKLIHOOD """
        logliklihood_tensor = policy(flat_states,flat_actions)

        """ CALCULATE ADVANTAGE (MCMC) """
        A_hat = -flat_cumsum.detach()

        """ CALCULATE POLICY GRADIENT OBJECTIVE """
        expectation = torch.dot(A_hat.reshape(-1),logliklihood_tensor.reshape(-1))/self.trajectory_length

        """ RETURN """
        return expectation/self.simulations

class GAME_SAMPLES(torch.nn.Module):

    def __init__(self, state_size, action_size, task, trajectory_length, sample_size):
        """ USED AS CONTAINER FOR TRAJECTORY BATCHES TO SAVE INITIALIZATION TIME """
        # initialize enviroment
        self.env = gym.make(task)
        # intialize batches of states, actions, rewards
        self.states_batch = torch.zeros((sample_size, trajectory_length, state_size))
        self.actions_batch = torch.zeros((sample_size, trajectory_length, action_size))
        self.rewards_batch = torch.zeros((sample_size, trajectory_length))
        self.cumsum_batch = torch.zeros((sample_size, trajectory_length))
        # set trajectory_length and samples
        self.trajectory_length = trajectory_length
        self.sample_size = sample_size
        # see if we need to produce an int / float / array
        self.action_type = type(self.env.action_space.sample())

    def handle_completion(self, time, sample):
        """ THIS HANDLES EARLY TERMINATION OF GAME """
        # set reward to the final reward state
        self.rewards_batch[sample,time:] = torch.stack([torch.tensor(0.) for _ in range(self.trajectory_length - time)])
        # set the state as the state that was finished in
        self.states_batch[sample,time:,:] = torch.stack([self.states_batch[sample,time-1,:] for _ in range(self.trajectory_length - time)])
        # generate random actions to regularize
        self.actions_batch[sample,time:,:] = torch.stack([torch.tensor([self.env.action_space.sample()]) for _ in range(self.trajectory_length - time)])

    def sample_game(self, env, policy):
        """ SAMPLE FROM GAME UNDER ENVIROMENT AND POLICY """
        # initialize enviroment
        current_state = self.env.reset()
        # set reversed indices becuase pytorch is dumb sometimes
        reverse_idx = [i for i in range(self.trajectory_length)][::-1]
        # iterate over samples
        for sample in range(self.sample_size):
            # iterate through full trajectories
            for t in range(self.trajectory_length):
                # set the current state and action
                self.states_batch[sample,t,:] = torch.tensor(current_state)
                self.actions_batch[sample,t,:] = policy.sample_action(self.states_batch[sample,t,:])
                # update stacked
                action = self.actions_batch[sample,t,:].int()[0].numpy()
                # take a step in the enviroment
                current_state, reward, done, info = self.env.step(action)
                # add the reward
                self.rewards_batch[sample,t] = reward
                # check done flag for
                if done:
                    # pass the enviroment on to handle_completion
                    self.handle_completion(t,sample)
                    # reset enviroment
                    observation = self.env.reset()
                    break
            # update cumsum
            self.cumsum_batch[sample,:] = torch.cumsum(self.rewards_batch[sample,reverse_idx], 0)[reverse_idx]
        # return game samples
        return self.states_batch, self.actions_batch, self.rewards_batch, self.cumsum_batch

class GPU_cartpole(torch.nn.Module):

    def __init__(self, task, iterations, sample_size, trajectory_length, batch_size = 1, num_workers = 2):
        super(GPU_cartpole, self).__init__()
        self.task = task
        self.iterations = iterations
        self.sample_size = sample_size
        self.trajectory_length = trajectory_length
        # DataLoader info
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_gym_task(self):

        """ TRAIN AGENT TO COMPLETE GYM TASK VIA BASELINE ADJUSTED POLICY GRADIENTS """
        # initialize env to get info
        env = gym.make(self.task)
        # get state and action dimensions from enviroment
        action_size = 1 # [len(env.action_space.sample()) if hasattr(env.action_space.sample(), "__len__") else 1][0]
        state_size = 4 #[len(env.reset()) if hasattr(env.reset(), "__len__") else 1][0]
        actions = 2 # env.action_space.n

        """ INITIALIZE PRIMARY INTERACTION ENVIROMENT """
        game = GAME_SAMPLES(state_size, action_size, self.task, self.trajectory_length, self.sample_size)
        # initial policy
        policy = POLICY(state_size, action_size, actions)

        """ INFO TRACKING """
        # loss tracking
        loss_per_iteration = torch.zeros((self.iterations))
        # time per iteration
        time_per_iteration = torch.zeros((self.iterations))

        """ INITIALIZE WHAT WE CAN ON THE GPU """
        my_device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        # add loss module
        cuda_pgloss = PG_LOSS(self.trajectory_length, self.sample_size).to(my_device)
        # add model
        cuda_policy = POLICY(state_size, action_size, actions).to(my_device)
        # initialize optimizer
        optimizer = torch.optim.Adam(cuda_policy.parameters(), lr=5e-3)

        """ TRAIN AGENT """
        for iter in range(self.iterations):

            """ SAMPLE FROM SIMULATOR """
            # time for project info
            start = time.time()
            # get new batch of trajectories from simulator
            state_total, action_total, reward_total, cumsum_total = game.sample_game(env, policy)

            """ RUN MULTIPLE UPDATES ON GPU USING MINI-BATCHES """
            # initialize pytorch dataset for efficient data loading
            data = torch.utils.data.TensorDataset(state_total, action_total.long(), \
                                                  reward_total, cumsum_total)
            # set data loader
            data_loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

            """ MINI-BATCH UPDATES """
            # update model on gpu following mini-batches
            for r, (state_batch, action_batch, reward_batch, cumsum_batch) in enumerate(data_loader):
                # move everything over to the GPU
                state_batch = state_batch.to(my_device, non_blocking=True)
                action_batch = action_batch.to(my_device, non_blocking=True)
                reward_batch = reward_batch.to(my_device, non_blocking=True)
                cumsum_batch = cumsum_batch.to(my_device, non_blocking=True)
                # compute loss
                loss = cuda_pgloss(cuda_policy, state_batch, action_batch, reward_batch, cumsum_batch)
                # zero the parameter gradients
                optimizer.zero_grad()
                # backprop through computation graph
                loss.backward()
                # step optimizer
                optimizer.step()

            """ UPDATE PARAMETERS FROM CUDA -> GLOBAL """
            policy.load_state_dict(deepcopy(cuda_policy.state_dict()))

            """ UPDATE DATA STORAGE """
            # update time
            end = time.time()
            time_per_iteration[iter] = end - start
            # update expected total reward
            expected_reward = torch.sum(cumsum_total[:,0]) / self.sample_size
            loss_per_iteration[iter] = expected_reward

            """ PRINT STATEMENTS """
            if iter % floor((self.iterations+1)/100) == 0:
                print("Expected Sum of rewards: " + str(expected_reward))
                print("Loss: " + str(loss))
                print('Time per Iteration: ' + str(end - start) + ' at iteration: ' + str(iter))
                print("Percent complete: " + str(floor(100*iter/self.iterations)))

        """ RETURN """
        # return the trained policy and the loss per iteration
        return policy, loss_per_iteration, time_per_iteration

def main():
    # """ IGNORE GYM WARNINGS """
    # gym.logger.set_level(60)

    """ GYM TASK 1 """
    # initialize hyper parameters
    task = 'CartPole-v0'
    iterations = 100
    sample_size = 800
    trajectory_length = 200
    # data loader info
    batch_size = 200
    workers = 6

    # set algorithm
    algorithm = GPU_cartpole(task, iterations, sample_size, trajectory_length, batch_size, workers)
    # train policy
    policy, loss_per_iteration, time_per_iteration = algorithm.train_gym_task()

    # plot loss
    plt.figure(1)
    plt.tight_layout()
    fig1 = plt.plot([i for i in range(len(loss_per_iteration))], loss_per_iteration.detach().numpy())
    plt.ylabel('Average Cumulative Reward')
    plt.xlabel('Iteration')
    plt.title(task)
    plt.savefig('loss_GPU.png')

    plt.figure(2)
    plt.tight_layout()
    fig2 = plt.plot([i for i in range(len(time_per_iteration))], time_per_iteration.detach().numpy())
    plt.ylabel('Evaluation Time Per Iteration')
    plt.xlabel('Iteration')
    plt.title(task)
    plt.savefig('time_GPU.png')

if __name__ == '__main__':
    main()
