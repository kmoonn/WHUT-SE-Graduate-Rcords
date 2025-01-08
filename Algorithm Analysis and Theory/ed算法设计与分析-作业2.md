# 第二章
![[Pasted image 20241102143932.png]]
## DDPG
### 核心思想

深度确定性策略梯度（Deep Deterministic Policy Gradient, DDPG）是一种用于处理连续动作空间的深度强化学习算法，主要结合了策略梯度和Q学习的思想。其核心思想如下：

1. **Actor-Critic架构**：DDPG使用两个主要组件——Actor和Critic。Actor负责生成动作，而Critic则评估该动作的价值。Actor的目标是最大化Critic的输出，从而提高策略的质量。

2. **确定性策略**：与其他强化学习算法（如DQN）不同，DDPG采用确定性策略，即给定状态直接输出动作。这使得DDPG在连续动作空间中表现优越。

3. **经验回放**：DDPG利用经验回放机制来存储过去的经历，从中随机采样进行训练。这减少了样本间的相关性，提高了训练的稳定性。

4. **目标网络**：为了解决Q值估计的不稳定性，DDPG使用目标网络（Actor和Critic的延迟副本）。这些目标网络在更新时使用较小的步长，这有助于平滑学习过程。

5. **探索机制**：DDPG引入了噪声（如Ornstein-Uhlenbeck噪声）来促进探索，保证在训练初期可以充分探索动作空间。

### 实验设计

为了评估DDPG及其改进算法的性能，可以设计以下实验：

1. **实验环境**：
   - 使用OpenAI Gym或Mujoco等模拟环境，选择一些适合连续动作空间的任务（如倒立摆、摆球等）。

2. **基线算法**：
   - 对比DDPG与其他算法（如PPO、TRPO、SAC等）的性能，以评估DDPG的优劣。

3. **超参数调优**：
   - 实验不同的超参数设置（学习率、批量大小、折扣因子等），观察对性能的影响。

4. **改进算法的实现**：
   - 实现DDPG的改进版本，如使用双Q网络、优先经验回放、改进的探索策略等。

5. **性能评估指标**：
   - 使用累积奖励、训练时间、成功率等指标来评估算法的性能。
   - 记录每个算法的学习曲线，以比较收敛速度和稳定性。
```python
import numpy as np
import gym
import random
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)

# DDPG参数
STATE_DIM = 3
ACTION_DIM = 1
ACTION_BOUND = 2  # 动作范围[-2, 2]
BUFFER_SIZE = 100000
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.005
LEARNING_RATE = 0.001

# 构建Actor网络
def create_actor():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(400, activation='relu', input_shape=(STATE_DIM,)),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(ACTION_DIM, activation='tanh')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))
    return model

# 构建Critic网络
def create_critic():
    state_input = tf.keras.Input(shape=(STATE_DIM,))
    action_input = tf.keras.Input(shape=(ACTION_DIM,))
    concat = tf.keras.layers.Concatenate()([state_input, action_input])
    x = tf.keras.layers.Dense(400, activation='relu')(concat)
    x = tf.keras.layers.Dense(300, activation='relu')(x)
    output = tf.keras.layers.Dense(1)(x)
    
    model = tf.keras.Model(inputs=[state_input, action_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))
    return model

# Ornstein-Uhlenbeck噪声
class OUNoise:
    def __init__(self, action_dim):
        self.mu = np.zeros(action_dim)
        self.theta = 0.15
        self.sigma = 0.2
        self.action_dim = action_dim
        self.state = self.mu.copy()
        
    def reset(self):
        self.state = self.mu.copy()
        
    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
        
    def add(self, experience):
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def size(self):
        return len(self.buffer)

# DDPG Agent
class DDPGAgent:
    def __init__(self):
        self.actor = create_actor()
        self.critic = create_critic()
        self.target_actor = create_actor()
        self.target_critic = create_critic()
        self.noise = OUNoise(ACTION_DIM)
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.update_target_networks()

    def update_target_networks(self):
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

    def act(self, state):
        action = self.actor.predict(state.reshape(1, STATE_DIM))
        action = action + self.noise.evolve_state()
        return np.clip(action, -ACTION_BOUND, ACTION_BOUND)

    def learn(self):
        if self.replay_buffer.size() < BATCH_SIZE:
            return
        batch = self.replay_buffer.sample(BATCH_SIZE)
        states, actions, rewards, next_states = zip(*batch)
        
        target_actions = self.target_actor.predict(np.array(next_states))
        target_qs = self.target_critic.predict([np.array(next_states), target_actions])
        targets = np.array(rewards) + GAMMA * target_qs.flatten()
        
        self.critic.train_on_batch([np.array(states), np.array(actions)], targets)
        
        actions_for_grad = self.actor.predict(np.array(states))
        with tf.GradientTape() as tape:
            q_values = self.critic([np.array(states), actions_for_grad])
        grads = tape.gradient(q_values, actions_for_grad)
        
        self.actor.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

        self.update_target_networks()

# 训练DDPG
def train_ddpg(episodes):
    env = gym.make('Pendulum-v1')
    agent = DDPGAgent()
    rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        agent.noise.reset()
        
        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.add((state, action, reward, next_state))
            agent.learn()
            state = next_state
            total_reward += reward
            if done:
                break
        
        rewards.append(total_reward)
        print(f'Episode {episode + 1}, Reward: {total_reward:.2f}')
    
    env.close()
    return rewards

# 绘制奖励曲线
def plot_rewards(rewards):
    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('DDPG Training Rewards')
    plt.show()

# 执行实验
if __name__ == '__main__':
    episodes = 200
    rewards = train_ddpg(episodes)
    plot_rewards(rewards)

```

### 性能分析

1. **收敛速度**：
   - 观察各算法在不同任务上的学习曲线，比较收敛速度。DDPG通常在简单任务上收敛较快，但在复杂任务上可能较慢。

2. **稳定性**：
   - 评估训练过程中奖励的波动性，DDPG的波动性可能比其他算法更大，特别是在复杂环境中。

3. **策略表现**：
   - 在训练完成后，使用学习到的策略在环境中进行评估，比较不同算法的实际表现。

4. **鲁棒性**：
   - 测试算法在不同环境和任务上的表现，DDPG在一些任务上可能表现不如其他先进算法，但在特定问题上具有优势。