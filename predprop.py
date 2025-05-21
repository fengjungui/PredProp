import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
# collections and random removed as they are not directly used by the final PPO script
# tqdm might use collections.deque internally, but it's not a direct dependency we need to import.

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# --- Helper Functions ---
def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().cpu().numpy() # Ensure it's on CPU and is numpy
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)

# --- Network Classes ---
class PolicyNetContinuous(nn.Module): # Renamed from Actor
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.sigma_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = 2.0 * torch.tanh(self.mu_head(x)) # Action range [-2, 2] for Pendulum
        sigma = F.softplus(self.sigma_head(x)) + 1e-5 
        return mu, sigma

class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ForwardModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, output_dim):
        super(ForwardModel, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, state, action):
        # In PPOContinuous.update, 'actions' is shaped to [batch_size, action_dim]
        # before being passed to this model. So, no further reshaping of 'action'
        # should be necessary here assuming state is [batch_size, state_dim].
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# --- PPO Class ---
class PPOContinuous(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, actor_lr, critic_lr, 
                 forward_model_lr, gamma, lmbda, epochs, eps, device):
        super(PPOContinuous, self).__init__()
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs # Renamed from K_epochs
        self.eps = eps       # Renamed from eps_clip
        self.device = device

        self.actor = PolicyNetContinuous(state_dim, action_dim, hidden_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.forward_model = ForwardModel(state_dim, action_dim, hidden_dim, state_dim).to(device) # output_dim = state_dim

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.forward_model_optimizer = torch.optim.Adam(self.forward_model.parameters(), lr=forward_model_lr)

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        mu, sigma = self.actor(state)
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        return action.cpu().numpy()[0] # Return as numpy array

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).view(-1, self.actor.mu_head.out_features).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)

        # Train Forward Model
        # Ensure actions has the correct shape for forward_model: (batch_size, action_dim)
        # self.actor.mu_head.out_features is action_dim
        current_actions_for_fm = actions.view(-1, self.actor.mu_head.out_features)
        predicted_next_states = self.forward_model(states, current_actions_for_fm)
        forward_model_loss = F.mse_loss(predicted_next_states, next_states)
        
        self.forward_model_optimizer.zero_grad()
        forward_model_loss.backward()
        self.forward_model_optimizer.step()

        # PPO Update
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta).to(self.device)
        
        mu, std = self.actor(states)
        action_dists_old = torch.distributions.Normal(mu.detach(), std.detach())
        # Ensure actions used for log_prob match the output of actor (action_dim)
        old_log_probs = action_dists_old.log_prob(actions.view(-1, self.actor.mu_head.out_features))

        for _ in range(self.epochs):
            mu, std = self.actor(states)
            action_dists_new = torch.distributions.Normal(mu, std)
            log_probs = action_dists_new.log_prob(actions.view(-1, self.actor.mu_head.out_features))
            ratio = torch.exp(log_probs - old_log_probs)
            
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            
            actor_loss.backward()
            critic_loss.backward()
            
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def save(self, path):
        # Simplified path handling for now
        torch.save(self.actor.state_dict(), path + "_actor.pth")
        torch.save(self.critic.state_dict(), path + "_critic.pth")
        torch.save(self.forward_model.state_dict(), path + "_forward_model.pth")
        print(f'Models saved to {path}_actor.pth, {path}_critic.pth, and {path}_forward_model.pth')

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + "_actor.pth", map_location=self.device))
        self.critic.load_state_dict(torch.load(path + "_critic.pth", map_location=self.device))
        self.forward_model.load_state_dict(torch.load(path + "_forward_model.pth", map_location=self.device))
        print(f'Models loaded from {path}_actor.pth, {path}_critic.pth, and {path}_forward_model.pth')

# --- Training Function ---
def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    max_reward = -np.inf # Track max reward
    for i in range(10): # Outer loop for progress bar
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state, info = env.reset()
                done = False
                truncated = False 
                while not (done or truncated):
                    action = agent.take_action(state)
                    next_state, reward, done, truncated, info = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                
                agent.update(transition_dict)
                return_list.append(episode_return)

                if episode_return > max_reward: # Save if new max reward
                    max_reward = episode_return
                    agent.save('ppo_pendulum_model_best')


                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

# --- Main Execution Block ---
if __name__ == "__main__":
    actor_lr = 1e-4
    critic_lr = 5e-3
    forward_model_lr = 1e-3 # Added forward_model_lr
    num_episodes = 2000
    hidden_dim = 128
    gamma = 0.9
    lmbda = 0.9 # GAE lambda
    epochs = 10 # PPO epochs
    eps = 0.2   # PPO clip epsilon

    env_name = 'Pendulum-v1'
    env = gym.make(env_name) #, render_mode="human"
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] # Pendulum action_dim is 1

    agent = PPOContinuous(state_dim, action_dim, hidden_dim, actor_lr, critic_lr, 
                          forward_model_lr, gamma, lmbda, epochs, eps, device)

    return_list = train_on_policy_agent(env, agent, num_episodes)

    # Save final model
    agent.save('ppo_pendulum_model_final')

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'PPO on {env_name}')
    mv_return = moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.savefig(f'ppo_{env_name}_forward_model_returns.png')
    # plt.show() # Commented out to prevent blocking in non-interactive environments

    # Render trained agent (optional)
    # print("\nRendering trained agent...")
    # for _ in range(5):
    #     state, info = env.reset()
    #     done = False
    #     truncated = False
    #     episode_reward = 0
    #     while not (done or truncated):
    #         action = agent.take_action(state)
    #         next_state, reward, done, truncated, info = env.step(action)
    #         state = next_state
    #         episode_reward += reward
    #         env.render()
    #     print(f"Rendered episode reward: {episode_reward}")

    env.close()
    print("Training complete and environment closed.")
    print(f"Max reward during training: {np.max(return_list)}") # Print max reward
    print(f"Final agent saved to ppo_pendulum_model_final_*.pth")
    print(f"Best agent saved to ppo_pendulum_model_best_*.pth")
