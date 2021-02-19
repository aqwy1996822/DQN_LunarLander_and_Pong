import gym
import glob
from agent import *
from config import *
import matplotlib.pyplot as plt
# 学习过程曲线
# rewards = np.load('rewards/LunarLander-v2_rewards.npy')
# plt.plot(rewards)
# plt.show()
# average = [np.mean(rewards[i - 100:i]) for i in range(100, len(rewards))]
# plt.plot(average)
# plt.show()

# 学习结果展示
env = gym.make(RAM_ENV_NAME)
agent = Agent(env.observation_space.shape[0], env.action_space.n, BATCH_SIZE, LEARNING_RATE, TAU, GAMMA, DEVICE, False)
#检查是否有历史训练记录
state_dict_file_list = glob.glob('weights/{}_*_weights.pth'.format(RAM_ENV_NAME))
state_dict_file_reward_list = []
for state_dict_file in state_dict_file_list:
    state_dict_file_reward_list.append(int(state_dict_file.replace('_weights.pth', '').replace('weights\\{}_'.format(RAM_ENV_NAME), '')))
state_dict_file_best = 'weights/{}_{}_weights.pth'.format(RAM_ENV_NAME, max(state_dict_file_reward_list))
print('最佳存档为', state_dict_file_best)
agent.Q_local.load_state_dict(torch.load(state_dict_file_best))

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