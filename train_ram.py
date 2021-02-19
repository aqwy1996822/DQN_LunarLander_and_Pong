import gym
from agent import *
from config import *
import glob

def train(env, agent, num_episode, eps_init, eps_decay, eps_min, max_t):
    rewards_log = []
    average_log = []
    eps = eps_init
    max_average = 0
    for i in range(1, 1 + num_episode):
        episodic_reward = 0#每轮的reward初始化
        done = False
        state = env.reset()
        t = 0
        while not done and t < max_t:
            # env.render()
            t += 1
            action = agent.act(state, eps)#走一步
            next_state, reward, done, _ = env.step(action)#读取状态，奖励，是否结束
            agent.memory.append((state, action, reward, next_state, done))#加入经验池
            if t % 4 == 0 and len(agent.memory) >= agent.bs:
                agent.learn()
                if t % C_target2local == 0:
                    agent.soft_update(agent.tau)
            state = next_state.copy()
            episodic_reward += reward

        rewards_log.append(episodic_reward)
        average_log.append(np.mean(rewards_log[-100:]))
        print('\rEpisode {}, Reward {:.3f}, Average Reward {:.3f}'.format(i, episodic_reward, average_log[-1]), end='')
        if i % 100 == 0:
            print()
            if average_log[-1] > max_average or max_average == 0:
                max_average = average_log[-1]
                torch.save(agent.Q_local.state_dict(), 'weights/{}_{}_weights.pth'.format(RAM_ENV_NAME, int(max_average)))
                np.save('rewards/{}_rewards.npy'.format(RAM_ENV_NAME), rewards_log)
        eps = max(eps * eps_decay, eps_min)
    return rewards_log


if __name__ == '__main__':
    env = gym.make(RAM_ENV_NAME)
    print('observation_space',env.observation_space)
    print('action_space',env.action_space)
    agent = Agent(env.observation_space.shape[0], env.action_space.n, BATCH_SIZE, LEARNING_RATE, TAU, GAMMA, DEVICE,
                  False)#创建训练的成员
    #检查是否有历史训练记录
    state_dict_file_list = glob.glob('weights/{}_*_weights.pth'.format(RAM_ENV_NAME))
    state_dict_file_reward_list = []
    if len(state_dict_file_list)>0:
        for state_dict_file in state_dict_file_list:
            state_dict_file_reward_list.append(int(state_dict_file.replace('_weights.pth', '').replace('weights\\{}_'.format(RAM_ENV_NAME), '')))
        state_dict_file_best = 'weights/{}_{}_weights.pth'.format(RAM_ENV_NAME, max(state_dict_file_reward_list))
        print('最佳存档为', state_dict_file_best)
        agent.Q_local.load_state_dict(torch.load(state_dict_file_best))
        agent.Q_target.load_state_dict(torch.load(state_dict_file_best))
        # EPS_INIT=0.05
    rewards_log = train(env, agent, RAM_NUM_EPISODE, EPS_INIT, EPS_DECAY, EPS_MIN, MAX_T)#训练

    # agent.Q_local.to('cpu')
    # torch.save(agent.Q_local.state_dict(), '{}_weights.pth'.format(RAM_ENV_NAME))
