import torch
from torch import optim
from torch import nn
import gym
import numpy as np


def save_model(policy, episode_num):
    PATH = "mypolicy.pth"
    torch.save(policy.state_dict(), PATH)
    f = open("episode_num.txt", "w")
    f.write(f"episode: {episode_num}")
    f.close()

class Reinforce:
    def __init__(self, policy, env, optimizer):
        self.policy = policy
        self.env = env
        self.optimizer = optimizer

    @staticmethod
    def compute_expected_cost(trajectory, gamma, baseline, p=0.98):
        """
        Compute the expected cost of this episode for gradient backprop
        
        :param trajectory: a list of 3-tuple of (reward: Float, policy_output_probs: torch.Tensor, action: Int)
        :param gamma: gamma
        :param baseline: a simple running mean baseline to be subtracted from the total discounted returns
        :return: a 2-tuple of torch.tensor([cost]) of this episode that allows backprop and updated baseline
        """
        T = len(trajectory)
        G_inverse = []
        G_prev = 0
        for i in range(T-1, -1, -1):
            reward = trajectory[i][0]
            G_new = reward + G_prev * gamma
            G_inverse.append(G_new)
            G_prev = G_new

        G = torch.tensor(G_inverse[::-1])

        mu = G.mean()
        sigma = G.std(unbiased=False)
        G = (G - baseline) / sigma
        baseline = p * baseline + (1-p) * mu

        cost = 0.0

        for i in range(T):
            policy_probs, ppidx = trajectory[i][1], trajectory[i][2]
            policy_value = policy_probs[ppidx]
            cost += (-1) * G[i] * torch.log(policy_value)

        return cost, baseline.item()

    def train(self, num_episodes, gamma):
        """
        train the policy using REINFORCE for specified number of episodes
        :param num_episodes: number of episodes to train for
        :param gamma: gamma
        :return: self
        """
        self.policy.train()

        baseline = 0

        running_loss = 0

        report_num = 100

        running_reward = 0
        
        for episode_i in range(num_episodes):
            ### YOUR CODE HERE ###
            self.optimizer.zero_grad()

            trajectory = self.generate_episode()
            loss, baseline = self.compute_expected_cost(trajectory, gamma, baseline)
            running_loss += loss
            reward, _, _ = list(zip(*trajectory))
            running_reward += sum(reward)

            loss.backward()
            self.optimizer.step()

            if (episode_i % report_num == 0):
                print(f"episode {episode_i}//{episode_num}: reward = {running_reward / report_num}, loss = {running_loss / report_num}")
                running_loss = 0
                running_reward = 0

        return self

    def generate_episode(self):
        """
        run the environment for 1 episode
        :return: whatever needed for training
        """

        init_state = self.env.reset()
        init_state = torch.from_numpy(init_state)

        trajectory = []

        done = False
        state = init_state

        while not done:
            state = state.float()
            action_list = self.policy(state)

            m = torch.distributions.Categorical(action_list)
            action = m.sample().item()

            s, r, done, info = self.env.step(action)
            trajectory.append((r, action_list, action))

            state = torch.from_numpy(s)

        return trajectory


class MyPolicy(nn.Module):
    def __init__(self):
        super(MyPolicy, self).__init__()
        ### YOUR CODE HERE AND REMOVE `pass` below ###
        self.linear1 = nn.Linear(8, 16) # num states
        self.linear2 = nn.Linear(16,16)
        self.linear3 = nn.Linear(16,16)
        self.linear4 = nn.Linear(16,4) # num actions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        ### YOUR CODE HERE AND REMOVE `pass` below ###
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.relu(out)
        out = self.linear4(out)
        out = self.softmax(out)
        return out
        
def init_weights(m):
    '''Reference:
    this function is copied from stack_overflow
    https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
    '''
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.0)


def save_model(model, optimizer, episodes):
    state = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "episodes": episodes}
    torch.save(state, f"model_tmp_{episodes}")
    print("saving model finished.")

def load_model(PATH, model, optimizer):
    state = torch.load(PATH)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print("loading finished. ")

if __name__ == '__main__':
    # define and train policy here
    
    # GLOBALS:
    LR = 0.0001
    episode_num = 1000
    gamma = 0.99
    
    # params for reinforce
    policy = MyPolicy()
    policy.apply(init_weights)
    
    env = gym.make('LunarLander-v2') # actions: 4, states: 8
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    load_model("model_tmp_3000", policy, optimizer)

    # reinforce class for training
    MyRL = Reinforce(policy, env, optimizer)
    MyRL.train(episode_num, gamma)

    save_model(MyRL.policy, MyRL.optimizer, episode_num+3000)

    PATH = "mypolicy.pth"
    # state = {'state_dict': policy.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(MyRL.policy.state_dict(), PATH)

    print("done")