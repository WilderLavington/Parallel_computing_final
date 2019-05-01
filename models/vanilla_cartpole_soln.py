import gym
import torch
from torch.distributions import MultivariateNormal, Categorical
import matplotlib.pyplot as plt
import torch.nn.functional as F
from numpy import floor
import time

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
        output = self.linear1(torch.FloatTensor(state))
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

class PG_LOSS(torch.nn.Module):
    """ POLICY GRADIENTS LOSS FUNCTION """
    def __init__(self, trajectory_length, simulations, decay):
        """ INITIALIZATIONS """
        super(PG_LOSS, self).__init__()
        self.simulations = simulations
        self.trajectory_length = trajectory_length
        # simple decaying baseline
        self.decay = decay # percentage to forget 0 -> 1 (1 = all new info)
        self.decay_baseline = torch.zeros(trajectory_length)

    def UpdateBaseline(self, new_sample):
        """ UPDATES THE APPROXIMATE BASELINE WITH NEW MCMC SAMPLE OF ROLLOUT """
        # compute simple baseline for each timestep
        new_baseline = torch.sum(new_sample,dim=1) / len(new_sample[0,:])
        # update the baseline
        if self.decay_baseline[0] == 0:
            self.decay_baseline = new_baseline
        else:
            self.decay_baseline = (1-self.decay)*self.decay_baseline + self.decay*new_baseline
        # dont return anthing
        return new_baseline

    def Advantage_estimator(self, logliklihood_tensor, trajectories_state_tensor, \
                            trajectories_action_tensor, trajectories_reward_tensor):
        """ COMPUTES ROLL OUT WITH MAX ENT REGULARIZATION """
        # initialize cumulative running average for states ahead
        cumulative_rollout = torch.zeros([self.trajectory_length, self.simulations])
        # calculate cumulative running average for states ahead
        cumulative_rollout[self.trajectory_length-1,:] = trajectories_reward_tensor[0, self.trajectory_length-1,:]
        # primary loop
        for time in reversed(range(0, self.trajectory_length-1)):
            cumulative_rollout[time,:] = trajectories_reward_tensor[0, time, :] \
                                         + cumulative_rollout[time+1,:]
        # update baseline
        new_baseline = self.UpdateBaseline(cumulative_rollout)
        # detach cumulative reward from computation graph
        advantage = cumulative_rollout.clone().detach()
        # return MCMC roll out values
        return advantage, new_baseline

    def forward(self, policy, trajectories_state_tensor, trajectories_action_tensor, trajectories_reward_tensor):
        """ CALCULATE LOG LIKLIHOOD OF TRAJECTORIES """
        # initialize tensor for log liklihood stuff
        logliklihood_tensor = torch.zeros([self.trajectory_length, self.simulations])
        # generate tensor for log liklihood stuff
        for simulation in range(self.simulations):
            for time in range(self.trajectory_length):
                # [simulation #, value, time step]
                logliklihood_tensor[time,simulation] = policy.logprob_action(trajectories_state_tensor[:,time,simulation],\
                                                                       trajectories_action_tensor[:,time,simulation])

        """ CALCULATE ADVANTAGE (MCMC) """
        A_hat, expected_reward = self.Advantage_estimator(logliklihood_tensor, trajectories_state_tensor, trajectories_action_tensor, \
                                         trajectories_reward_tensor)

        """ CALCULATE POLICY GRADIENT OBJECTIVE """
        # initialize expectation tensor
        expectation = 0
        # calculate instance of expectation for timestep then calc sample mean
        for time in range(self.trajectory_length):
            expectation += torch.dot(A_hat[time,:], logliklihood_tensor[time,:])/self.simulations
        # sum accross time
        expectation = torch.sum(expectation)/self.trajectory_length

        """ RETURN """
        return -expectation, expected_reward

class GAME_SAMPLES(torch.nn.Module):

    def __init__(self, state_size, action_size, task, trajectory_length, sample_size):
        """ USED AS CONTAINER FOR TRAJECTORY BATCHES TO SAVE INITIALIZATION TIME """
        # initialize enviroment
        self.env = gym.make(task)
        # intialize batches of states, actions, rewards
        self.states_batch = torch.zeros((state_size, trajectory_length, sample_size))
        self.actions_batch = torch.zeros((action_size, trajectory_length, sample_size))
        self.rewards_batch = torch.zeros((1, trajectory_length,sample_size))
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
        # iterate over samples
        for sample in range(self.sample_size):
            # iterate through full trajectories
            for t in range(self.trajectory_length):
                # set the current state and action
                self.states_batch[:,t,sample] = torch.tensor(current_state)
                self.actions_batch[:,t,sample] = policy.sample_action(current_state)
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
        # return game samples
        return self.states_batch, self.actions_batch, self.rewards_batch

def train_gym_task(task, iterations, sample_size, trajectory_length, baseline_decay):
    """ TRAIN AGENT TO COMPLETE GYM TASK VIA BASELINE ADJUSTED POLICY GRADIENTS """

    # initialize env to get info
    env = gym.make(task)

    # get state and action dimensions from enviroment
    action_size = 1 # [len(env.action_space.sample()) if hasattr(env.action_space.sample(), "__len__") else 1][0]
    state_size = 4 #[len(env.reset()) if hasattr(env.reset(), "__len__") else 1][0]
    actions = 2 # env.action_space.n

    # initialize personal enviroment
    game = GAME_SAMPLES(state_size, action_size, task, trajectory_length, sample_size)
    # initial policy
    policy = POLICY(state_size, action_size, actions)
    # initialize loss function
    Loss_fxn = PG_LOSS(trajectory_length, sample_size, baseline_decay)
    # initialize optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    # loss tracking
    loss_per_iteration = torch.zeros((iterations))
    time_per_iteration = torch.zeros((iterations))
    """ TRAIN AGENT """
    for iter in range(iterations):
        # time setup
        start = time.time()
        # get new batch of trajectories from simulator
        states_batch, actions_batch, rewards_batch = game.sample_game(env, policy)
        # compute loss
        loss, expected_reward = Loss_fxn(policy, states_batch, actions_batch, rewards_batch)
        # zero the parameter gradients
        optimizer.zero_grad()
        # backprop through computation graph
        loss.backward()
        # step optimizer
        optimizer.step()
        # update loss
        loss_per_iteration[iter] = expected_reward[0]
        # print statements
        if iter % floor((iterations+1)/100) == 0:
            print("Expected Sum of rewards: " + str(expected_reward[0]))
            print("Loss: " + str(loss))
            end = time.time()
            print("Time per Iteration: " + str(end - start))
            print("Percent complete: " + str(floor(100*iter/iterations)))
        # update time
        end = time.time()
        time_per_iteration[iter] = end - start
    # return the trained policy and the loss per iteration
    return policy, loss_per_iteration, time_per_iteration

def main():
    # """ IGNORE GYM WARNINGS """
    # gym.logger.set_level(40)

    """ GYM TASK 1 """
    # initialize hyper parameters
    task = 'CartPole-v0'
    iterations = 500
    sample_size = 200
    trajectory_length = 200
    baseline_decay = 0.5

    # train policy
    policy, loss_per_iteration, time_per_iteration = train_gym_task(task, iterations, sample_size, trajectory_length, baseline_decay)

    # plot loss
    plt.figure(1)
    plt.tight_layout()
    fig1 = plt.plot([i for i in range(len(loss_per_iteration))], loss_per_iteration.detach().numpy())
    plt.ylabel('Average Cumulative Reward')
    plt.xlabel('Iteration')
    plt.title(task)
    plt.savefig('loss_vanilla.png')

    plt.figure(2)
    plt.tight_layout()
    fig2 = plt.plot([i for i in range(len(time_per_iteration))], time_per_iteration.detach().numpy())
    plt.ylabel('Evaluation Time Per Iteration')
    plt.xlabel('Iteration')
    plt.title(task)
    plt.savefig('time_vanilla.png')

if __name__ == '__main__':
    main()
