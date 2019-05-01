import gym
import torch
from torch.distributions import MultivariateNormal, Categorical
import matplotlib.pyplot as plt
import torch.nn.functional as F
from numpy import floor
import torch.multiprocessing as mp
from copy import deepcopy
import time


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
                        cumulative_rollout, logliklihood_tensor):

        """ CALCULATE LOG LIKLIHOOD OF TRAJECTORIES """
        # initialize tensor for log liklihood stuff
        logliklihood_tensor = 0*logliklihood_tensor

        """ CALCULATE LIKLIHOOD """
        logliklihood_tensor = policy(state_tensor,action_tensor)

        """ CALCULATE ADVANTAGE (MCMC) """
        A_hat = -cumulative_rollout.detach()

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
        self.states_batch = torch.zeros((state_size, trajectory_length, sample_size))
        self.actions_batch = torch.zeros((action_size, trajectory_length, sample_size))
        self.rewards_batch = torch.zeros((1, trajectory_length, sample_size))
        # initialize stacked versions of tensors
        self.stacked_states = torch.zeros((state_size, trajectory_length*sample_size))
        self.stacked_actions = torch.zeros((action_size, trajectory_length*sample_size),dtype=torch.long)
        self.stacked_rewards = torch.zeros((1, trajectory_length*sample_size))
        self.stacked_cumsum = torch.zeros((1, trajectory_length*sample_size))
        # set trajectory_length and samples
        self.trajectory_length = trajectory_length
        self.sample_size = sample_size
        # see if we need to produce an int / float / array
        self.action_type = type(self.env.action_space.sample())

    def handle_completion(self, time, sample):
        """ THIS HANDLES EARLY TERMINATION OF GAME """
        # set reward to the final reward state
        self.rewards_batch[0,time:,sample] = torch.stack([torch.tensor(0.) for _ in range(self.trajectory_length - time)])
        # set the state as the state that was finished in
        self.states_batch[:,time:,sample] = torch.stack([self.states_batch[:,time-1,sample] for _ in range(self.trajectory_length - time)]).transpose(0, 1)
        # generate random actions to regularize
        self.actions_batch[:,time:,sample] = torch.stack([torch.tensor([self.env.action_space.sample()]) for _ in range(self.trajectory_length - time)]).transpose(0, 1)

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
                self.states_batch[:,t,sample] = torch.tensor(current_state)
                self.actions_batch[:,t,sample] = policy.sample_action(current_state)
                # update stacked
                action = self.actions_batch[:,t,sample].int()[0].numpy()
                # take a step in the enviroment
                current_state, reward, done, info = self.env.step(action)
                # add the reward
                self.rewards_batch[0,t,sample] = reward
                # check done flag for
                if done:
                    # pass the enviroment on to handle_completion
                    self.handle_completion(t,sample)
                    # reset enviroment
                    observation = self.env.reset()
                    break
            # update stacked variables
            self.stacked_states[:,self.trajectory_length*sample:self.trajectory_length*(sample+1)] = self.states_batch[:,:,sample]
            self.stacked_actions[:,self.trajectory_length*sample:self.trajectory_length*(sample+1)] = self.actions_batch[:,:,sample]
            self.stacked_rewards[:,self.trajectory_length*sample:self.trajectory_length*(sample+1)] = self.rewards_batch[0,:,sample]
            self.stacked_cumsum[:,self.trajectory_length*sample:self.trajectory_length*(sample+1)] = torch.cumsum(self.rewards_batch[0,reverse_idx,sample], 0)[reverse_idx]

        # return game samples
        return self.states_batch, self.actions_batch, self.rewards_batch, \
            self.stacked_states, self.stacked_actions, self.stacked_rewards, self.stacked_cumsum

class vectorized_cartpole(torch.nn.Module):

    def __init__(self, task, iterations, sample_size, trajectory_length):
        super(vectorized_cartpole, self).__init__()
        self.task = task
        self.iterations = iterations
        self.sample_size = sample_size
        self.trajectory_length = trajectory_length
        # approximate expected_return
        self.total = self.sample_size*self.trajectory_length
        self.stride = [i for i in range(self.total) if i % self.trajectory_length == 0]

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
        # add loss module
        pgloss = PG_LOSS(self.trajectory_length, self.sample_size)
        # initialize optimizer
        optimizer = torch.optim.Adam(policy.parameters(), lr=5e-3)
        # initialize other tensors that will be used
        loglik_batch = torch.zeros([1, self.trajectory_length*self.sample_size])

        """ INFO TRACKING """
        # loss tracking
        loss_per_iteration = torch.zeros((self.iterations))
        # time per iteration
        time_per_iteration = torch.zeros((self.iterations))

        """ TRAIN AGENT """
        for iter in range(self.iterations):
            # time for project info
            start = time.time()
            # get new batch of trajectories from simulator
            _, _, _, stacked_state, stacked_action, stacked_reward, stacked_cumsum = game.sample_game(env, policy)
            # compute loss
            loss = pgloss(policy, stacked_state.t(), stacked_action.t(), stacked_reward, stacked_cumsum, loglik_batch)
            # zero the parameter gradients
            optimizer.zero_grad()
            # backprop through computation graph
            loss.backward()
            # step optimizer
            optimizer.step()
            # update time
            end = time.time()
            time_per_iteration[iter] = end - start
            # update expected total reward
            expected_reward = torch.sum(stacked_cumsum[0,self.stride]) / self.sample_size
            loss_per_iteration[iter] = expected_reward
            # print statements
            if iter % floor((self.iterations+1)/100) == 0:
                print("Expected Sum of rewards: " + str(expected_reward))
                print("Loss: " + str(loss))
                print("Time per Iteration: " + str(end - start))
                print("Percent complete: " + str(floor(100*iter/self.iterations)))

        # return the trained policy and the loss per iteration
        return policy, loss_per_iteration, time_per_iteration

def main():
    # """ IGNORE GYM WARNINGS """
    # gym.logger.set_level(40)

    """ GYM TASK 1 """

    # initialize hyper parameters
    task = 'CartPole-v0'
    iterations = 100
    sample_size = 200
    trajectory_length = 200

    # set algorithm
    algorithm = vectorized_cartpole(task, iterations, sample_size, trajectory_length)
    # train policy
    policy, loss_per_iteration, time_per_iteration = algorithm.train_gym_task()

    # plot loss
    plt.figure(1)
    plt.tight_layout()
    fig1 = plt.plot([i for i in range(len(loss_per_iteration))], loss_per_iteration.detach().numpy())
    plt.ylabel('Average Cumulative Reward')
    plt.xlabel('Iteration')
    plt.title(task)
    plt.savefig('loss_vectorized.png')

    plt.figure(2)
    plt.tight_layout()
    fig2 = plt.plot([i for i in range(len(time_per_iteration))], time_per_iteration.detach().numpy())
    plt.ylabel('Evaluation Time Per Iteration')
    plt.xlabel('Iteration')
    plt.title(task)
    plt.savefig('time_vectorized.png')

if __name__ == '__main__':
    main()
