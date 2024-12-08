# _*_ coding : utf-8 _*_
# @Time : 2024/11/2 下午2:46
# @Author : Kmoon_Hs
# @File : main

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
