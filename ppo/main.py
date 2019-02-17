# Copyright John&Bill, Inc., 2019.

"""Implement the PPO model.

Currently implemented on the Cart-Pole environment.
https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson
    Observation: 
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24 deg        24 deg
        3	Pole Velocity At Tip      -Inf            Inf
        
    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right
        
        Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Pole Angle is more than 12 degrees
        Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """

import logging
import random
import time

import gym
import numpy as np
import tensorflow as tf

logging.basicConfig(level=logging.INFO, format='%(message)s')

class PPO:

    def __init__(self, discount_rate, batch_size, epsilon, learning_rate, render_frequency):
        self.env = gym.make("Pendulum-v0")
        self.observation_dimension = self.env.observation_space.shape[0]
        # self.action_dimension = self.env.action_space.n
        self.action_dimension = 9
        self.sum_steps = 0
        self.max_steps = self.env.spec.max_episode_steps
        self.num_episodes = 0
        self.discount_rate = discount_rate
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.render_frequency = render_frequency
        self.init_model()

    def init_model(self):
        """Define the model and loss."""
        with tf.variable_scope('model'):
            self.observations = tf.placeholder(shape=(None, self.observation_dimension),
                                               dtype=tf.float32)
            hidden_layer = tf.layers.dense(self.observations, units=32, activation='tanh')
            self.policy_probabilities = tf.layers.dense(hidden_layer,
                                                        units=self.action_dimension,
                                                        activation='softmax')
            self.old_policy_probabilities = tf.placeholder(shape=(None, self.action_dimension),
                                                           dtype=tf.float32)
            self.state_value = tf.layers.dense(hidden_layer, units=1, activation=None)
            self.action_taken = tf.placeholder(shape=(None,), dtype=tf.int32)
            # Set the return of the action that wasn't taken to 0.
            self.return_target = tf.placeholder(shape=(None, 1), dtype=tf.float32)

        action_taken_one_hot = tf.one_hot(indices=self.action_taken, depth=self.action_dimension)
        advantage = tf.subtract(self.return_target, self.state_value)
        advantage_by_action = tf.multiply(action_taken_one_hot, advantage)

        # Minimize the negative loss, which is equivalent to maximizing the return.
        policy_ratio = tf.realdiv(self.policy_probabilities, self.old_policy_probabilities)
        unclipped_weighted_advantage = tf.reduce_sum(tf.multiply(policy_ratio, advantage_by_action))
        clipped_policy_ratio = tf.clip_by_value(policy_ratio, 1 - self.epsilon, 1 + self.epsilon)
        clipped_weighted_advantage = tf.reduce_sum(tf.multiply(clipped_policy_ratio, 
                                                               advantage_by_action))
        assert(unclipped_weighted_advantage.shape == clipped_weighted_advantage.shape)
        min_weighted_advantage = tf.minimum(unclipped_weighted_advantage, clipped_weighted_advantage)
        policy_loss = tf.multiply(-1.0, min_weighted_advantage)
        value_loss = tf.losses.mean_squared_error(labels=self.return_target,
                                                  predictions=self.state_value)
        combined_loss = tf.add(policy_loss, value_loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_operation = optimizer.minimize(combined_loss)
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def get_action(self, observation):
        probs = self.session.run(self.policy_probabilities, feed_dict={
            self.observations:observation.reshape(1, -1)})[0]
        action_index = np.random.choice(self.action_dimension, size=1, p=probs)[0]
        return action_index, probs

    def human_policy(self, observation):
        thingy = observation[1] * -0.05 + observation[2]
        if thingy > 0:
            return 1
        else:
            return 0

    def log_vars(self):
        varses = tf.trainable_variables()
        vars_vals = self.session.run(varses)        
        for var, val in zip(varses, vars_vals):
            logging.info('var: {}, val: {}'.format(var, val))

    def discounted_returns(self, rewards, initial_return):
        discounted_return = initial_return
        discounted_returns = []
        for reward in reversed(rewards):
            # Because we're going backwards, we can do this trick:
            discounted_return = self.discount_rate * discounted_return + reward
            discounted_returns.append(discounted_return)
        return list(reversed(discounted_returns))
        
    def train_model(self, observations, actions, discounted_returns, old_policy_probabilities,
                    verbose=False):
        if verbose:
            logging.debug('obs: {}, rew: {}'.format(observations, discounted_returns))

        length = len(observations)
        assert(length == len(actions))
        assert(length == len(discounted_returns))
        returns_array = np.array(discounted_returns)
        observations_array = np.array(observations)
        self.session.run(self.train_operation, feed_dict={
            self.observations: observations_array, 
            self.return_target: returns_array.reshape(-1, 1),
            self.action_taken: np.array(actions),
            self.old_policy_probabilities: np.array(old_policy_probabilities)
        })

    def run_batch(self):
        total_observations = []
        total_actions = []
        total_discounted_rewards = []
        episode_rewards = []
        steps = []
        total_policy_probabilities = []
        while True:
            obs, act, reward, step, policy_probs = self.sample_episode()
            total_observations += obs
            total_actions += act
            total_discounted_rewards += reward
            episode_rewards.append(reward[0])
            steps.append(step)
            total_policy_probabilities += policy_probs
            if (len(total_observations) >= self.batch_size):
                break
        
        # Train and print info.
        self.train_model(total_observations, total_actions, total_discounted_rewards, 
                         total_policy_probabilities, verbose=True)
        # self.log_vars()
        # Log the average number of steps to complete the episode.
        logging.info('{} episodes complete. Average episode length: {}, average reward: {}'
                        .format(self.num_episodes, np.mean(np.array(steps)), np.mean(np.array(episode_rewards))))

    def sample_episode(self):
        self.num_episodes += 1
        observations, actions, rewards, policy_probabilities = [], [], [], []
        observation = self.env.reset()
        for step in range(1, self.max_steps + 1):
            if self.num_episodes % self.render_frequency == 0:
                self.env.render()

            observations.append(observation)
            action_index, probs = self.get_action(observation)
            action = -2.0 + (4 / (self.action_dimension - 1)) * action_index
            actions.append(action_index)
            policy_probabilities.append(probs)
            # The last variable, info, is always {}.
            observation, reward, done, _ = self.env.step(np.array([action]))
            rewards.append(reward)
            logging.debug('action: {}, obs: {}, rew: {}, done: {}'
                         .format(action, observation, reward, done))
            self.sum_steps += 1
            if done:
                break
        is_terminal = step == self.max_steps
        if is_terminal:
            initial_return = 0
        else:
            initial_return = self.session.run(self.state_value, feed_dict={
                self.observations: observation.reshape(1, -1)
            })[0][0]
        discounted_returns = self.discounted_returns(rewards, initial_return)
        return observations, actions, discounted_returns, step, policy_probabilities


def main():
    pg = PPO(discount_rate=1.0, batch_size=5000, epsilon=0.2, learning_rate=1e-2, render_frequency=1000)
    while True:
        pg.run_batch()

if __name__ == '__main__':
    main()
