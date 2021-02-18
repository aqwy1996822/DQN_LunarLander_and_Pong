import gym
from agent import *
from config import *
import matplotlib.pyplot as plt
# 学习过程曲线
rewards = np.load('LunarLander-v2_rewards.npy')
plt.plot(rewards)
plt.show()
average = [np.mean(rewards[i - 100:i]) for i in range(100, len(rewards))]
plt.plot(average)
plt.show()

# 学习结果展示

env = gym.make(RAM_ENV_NAME)
agent = Agent(env.observation_space.shape[0], env.action_space.n, BATCH_SIZE, LEARNING_RATE, TAU, GAMMA, DEVICE, False)
agent.Q_local.load_state_dict(torch.load('LunarLander-v2_weights.pth'))

rewards_log = []
eps = EPS_MIN
num_episode = 10
max_t = 1000

for i in range(1, 1 + num_episode):
    episodic_reward = 0
    done = False
    state = env.reset()
    t = 0
    while not done and t < max_t:
        env.render()
        t += 1
        action = agent.act(state, eps)
        next_state, reward, done, _ = env.step(action)
        state = next_state.copy()
        episodic_reward += reward
    rewards_log.append(episodic_reward)
print(rewards_log)