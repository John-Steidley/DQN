# Copyright John&Bill, Inc., 2019.

"""Implement the REINFORCE model.

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

class REINFORCE:

    def __init__(self, discount_rate, batch_size, render_frequency):
        self.env = gym.make("CartPole-v1")
        self.init_model()
        self.sum_steps = 0
        self.max_steps = 500
        self.num_episodes = 0
        self.render_frequency = render_frequency
        self.discount_rate = discount_rate
        self.batch_size = batch_size

    def init_model(self):
        """Define the model and loss."""
        observation_dimension = self.env.observation_space.shape[0]
        action_dimension = self.env.action_space.n
        with tf.variable_scope('model'):
            self.observation_placeholder = tf.placeholder(shape=(None, observation_dimension), dtype=tf.float32)
            hidden_layer = tf.layers.dense(self.observation_placeholder, units=32, activation='tanh')
            self.policy_probabilities = tf.layers.dense(hidden_layer, units=action_dimension, activation='softmax')
            self.state_value = tf.layers.dense(hidden_layer, units=1, activation=None)
            # Set the return of the action that wasn't taken to 0.
            self.policy_target = tf.placeholder(shape=(None, 2), dtype=tf.float32)
            self.value_target = tf.placeholder(shape=(None, 1), dtype=tf.float32)
            
        # Minimize the negative loss, which is equivalent to maximizing the return.
        # TODO() add discount factor as suggested in sutton/bartow
        policy_loss = tf.multiply(-1.0, tf.multiply(tf.log(self.policy_probabilities), self.policy_target))
        value_loss = tf.losses.mean_squared_error(labels=self.value_target, predictions=self.state_value)
        combined_loss = tf.add(tf.reduce_sum(policy_loss), value_loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=5e-3)
        self.train_operation = optimizer.minimize(combined_loss)
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())
        

    def get_action(self, observation):
        # TODO() This depends on having only two choices for actions
        probs = self.session.run(self.policy_probabilities, feed_dict={self.observation_placeholder: observation.reshape(1, -1)})[0]
        prob_zero, prob_one = probs
        rand_number = random.random()
        if rand_number < prob_zero:
            return 0
        else:
            return 1

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
        return reversed(discounted_returns)
        
    def train_model(self, observations, actions, discounted_returns, verbose=False):
        if verbose:
            logging.debug('obs: {}, rew: {}'.format(observations, discounted_returns))

        length = len(observations)
        assert(length == len(actions))
        assert(length == len(discounted_returns))
        returns_array = np.array(discounted_returns)
        stacked_returns_array = np.column_stack((returns_array, returns_array))
        assert(stacked_returns_array.shape == (length, 2))

        actions_array = np.array(actions)
        actions_inverted = 1 - actions_array
        actions_one_hot_array = np.column_stack((actions_inverted, actions_array))
        assert(actions_one_hot_array.shape == (length, 2))

        policy_target = np.multiply(actions_one_hot_array, stacked_returns_array)        
        observations_array = np.array(observations)
        self.session.run(self.train_operation, feed_dict={
            self.observation_placeholder: observations_array, 
            self.policy_target: policy_target,
            self.value_target: returns_array.reshape(-1, 1)
        })

    def run_batch(self):
        total_observations, total_actions, total_discounted_rewards, steps = [], [], [], []
        while True:
            observations, actions, discounted_rewards, step = self.sample_episode()
            total_observations += observations
            total_actions += actions
            total_discounted_rewards += discounted_rewards
            steps.append(step)
            if (len(total_observations) >= self.batch_size):
                break
        
        # Train and print info.
        self.train_model(total_observations, total_actions, total_discounted_rewards, verbose=True)
        # self.log_vars()
        # Log the average number of steps to complete the episode.
        logging.info('{} episodes complete. Average episode length: {}'
                        .format(self.num_episodes, np.mean(np.array(steps))))

    def sample_episode(self):
        self.num_episodes += 1
        observations, actions, rewards = [], [], []
        observation = self.env.reset()
        for step in range(1, self.max_steps + 1):
            if self.num_episodes % self.render_frequency == 0:
                self.env.render()

            observations.append(observation)
            action = self.get_action(observation)
            actions.append(action)
            # The last variable, info, is always {}.
            observation, reward, done, _ = self.env.step(action)
            rewards.append(reward)
            succeeded = step >= self.max_steps
            logging.debug('action: {}, obs: {}, rew: {}, done: {}'
                         .format(action, observation, reward, done))
            self.sum_steps += 1
            if done:
                break
        is_terminal = step == self.max_steps
        if is_terminal:
            initial_return = 0
        else:
            initial_return = self.session.run(self.state_value, feed_dict: {
                self.observation_placeholder: observation.reshape(1, -1)
            })
        discounted_returns = self.discounted_returns(rewards, initial_return)
        return observations, actions, discounted_returns, step


def main():
    pg = REINFORCE(discount_rate=1.0, batch_size=5000, render_frequency=1000)
    while True:
        pg.run_batch()

if __name__ == '__main__':
    main()
