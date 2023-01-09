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
        self.sarsa_table = np.zeros(self.buckets + (self.environment.action_space.n,))

        self.episode = 0
        self.min_lr = 0.05
        self.min_epsilon = 0.05
        self.discount = 0.95
        self.decay = 25

    def learn(self, max_attempts, *, save: bool = False):
        with open('results/s_result_04_4.csv', 'w') as f:
            f.write(f"idx,lr,epsilon,reward_sum,discount,decay\n")
            for idx in range(max_attempts):
                self.episode += 1
                reward_sum = self.attempt()
                if save:
                    f.write(f"{idx},{self.min_lr},{self.min_epsilon},{reward_sum},{self.discount},{self.decay}\n")
                else:
                    print(f"{idx} {reward_sum}")
        print('done')

    def attempt(self):
        observation = self.discretize(self.environment.reset()[0])
        done = False
        reward_sum = 0.0
        while not done:
            action = self.pick_action(observation)
            new_observation, reward, done, _, info = self.environment.step(action)
            new_observation = self.discretize(new_observation)
            new_action = self.pick_action(observation)
            self.update_sarsa(action, observation, new_observation, new_action, reward)
            observation = new_observation
            reward_sum += reward
        self.attempt_no += 1
        return reward_sum

    def discretize(self, observation):
        discretized = list()
        for i in range(len(observation)):
            scaling = (observation[i] + abs(self.lower_bounds[i])) / (self.upper_bounds[i] - self.lower_bounds[i])
            new_obs = int(round((self.buckets[i] - 1) * scaling))
            new_obs = min(self.buckets[i] - 1, max(0, new_obs))
            discretized.append(new_obs)
        return tuple(discretized)

    def pick_action(self, observation):
        if np.random.random() < self.epsilon:
            return self.environment.action_space.sample()
        return np.argmax(self.sarsa_table[observation])

    def update_sarsa(self, action, observation, new_observation, new_action, reward):
        self.sarsa_table[observation][action] += self.learning_rate * \
                                                 (reward + self.discount * (self.sarsa_table[new_observation][new_action]
                                                                            - self.sarsa_table[observation][action]))

        
    @property
    def epsilon(self):
        return max(self.min_epsilon, min(1., 1. - math.log10((self.episode + 1) / self.decay)))

    @property
    def learning_rate(self):
        return max(self.min_lr, min(1., 1. - math.log10((self.episode + 1) / self.decay)))


def main():
    learner = QLearner()
    learner.learn(5000, save=True)


if __name__ == '__main__':
    main()
