import math

import gym
import numpy as np


class QLearner:
    def __init__(self):
        # self.environment = gym.make('CartPole-v1', render_mode="human")
        self.environment = gym.make('CartPole-v1')
        self.attempt_no = 1
        self.upper_bounds = [
            self.environment.observation_space.high[0],
            0.5,
            self.environment.observation_space.high[2],
            math.radians(50)
        ]
        self.lower_bounds = [
            self.environment.observation_space.low[0],
            -0.5,
            self.environment.observation_space.low[2],
            -math.radians(50)
        ]

        self.buckets = (1, 1, 6, 12)
        self.q_table = np.zeros(self.buckets + (self.environment.action_space.n,))

        self.episode = 0
        self.min_lr = 0.1
        self.min_epsilon = 0.1
        self.discount = 1.0
        self.decay = 25

    def learn(self, max_attempts):
        for idx in range(max_attempts):
            self.episode += 1
            reward_sum = self.attempt()
            print(f"{idx} {reward_sum}")

    def attempt(self):
        observation = self.discretise(self.environment.reset()[0])
        done = False
        reward_sum = 0.0
        while not done:
            # self.environment.render()
            action = self.pick_action(observation)
            new_observation, reward, done, _, info = self.environment.step(action)
            new_observation = self.discretise(new_observation)
            self.update_knowledge(action, observation, new_observation, reward)
            observation = new_observation
            reward_sum += reward
        self.attempt_no += 1
        return reward_sum

    def discretise(self, observation):
        # bins = np.linspace(self.lower_bounds, self.upper_bounds, 5)
        # obs = np.digitize(observation, bins)
        #
        # for idx in

        discretized = list()
        for i in range(len(observation)):
            scaling = (observation[i] + abs(self.lower_bounds[i])) / (self.upper_bounds[i] - self.lower_bounds[i])
            new_obs = int(round((self.buckets[i] - 1) * scaling))
            new_obs = min(self.buckets[i] - 1, max(0, new_obs))
            discretized.append(new_obs)
        return tuple(discretized)
        # return tuple(obs)

    def pick_action(self, observation):
        if np.random.random() < self.epsilon:
            return self.environment.action_space.sample()
        return np.argmax(self.q_table[observation])

    def update_knowledge(self, action, observation, new_observation, reward):
        self.q_table[observation][action] += self.learning_rate * (
                reward + self.discount * np.max(self.q_table[new_observation]) - self.q_table[observation][action])

    @property
    def epsilon(self):
        return max(self.min_epsilon, min(1., 1. - math.log10((self.episode + 1) / self.decay)))

    @property
    def learning_rate(self):
        return max(self.min_lr, min(1., 1. - math.log10((self.episode + 1) / self.decay)))


def main():
    learner = QLearner()
    learner.learn(10000)


if __name__ == '__main__':
    main()
