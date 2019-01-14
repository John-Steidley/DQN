# Copyright John&Bill, Inc., 2018.

"""Implement the DQN model.

Currently implemented on the Mountain Car V-0 environment.
https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py

Actions:
    0 is accelerate backwards
    1 is do nothing
    2 is accelerate forward

State:
    position: -1.2 - .6
              The bottom of the valley is at position -pi/6 ~= -0.52359.
              The top of the hill w/ the goal is at position pi/6 ~= 0.52359.
    velocity: -.07 - .07

"""
from collections import namedtuple
import logging
import operator
import random

import gym
import numpy as np
import tensorflow as tf

# things that are different
# layer norm
# (FIXED) eps decay
# amount of data before training
# lr decay?
# They use Adam, we use vanilla SGD
# (FIXED) They had 64 neurons and Tanh activation (other non-mountaincar use 2 hidden layers)
# Size of replay buffer

# Note: can change to level=logging.info to eliminate most logs
logging.basicConfig(level=logging.DEBUG, format='%(message)s')

# Experiences allow us to train the model on transitions from a broader
# behavior distribution.
Experience = namedtuple('Experience', [
    'old_observation',
    'action',
    'reward',
    'new_observation',
    'found_goal'
])


def init_model():
    """Create a neural network to predict the action-values for each action
    given the observations (position, velocity).
    """
    # Weights are initialized using glorot_normal by default.
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='tanh', input_shape=(2,)),
        tf.keras.layers.Dense(3)
    ])

    # After 100 episodes, have lr of .001.
    # lr_decay = .009/20000 # which is 4.5e-7
    optimizer = tf.keras.optimizers.SGD(lr=1e-3)
    # TODO: Consider switching to RMSProp to match the paper
    # optimizer = tf.keras.optimizers.RMSprop()
    # Defaults to lr=0.001, rho=0.9, epsilon=None, decay=0.0
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


def human_policy(observation):
    """Return the action that continues motion in the direction of velocity.

    Probably not quite optimal, but quite good. Achieves ~ 129.9 avg reward.
    """
    velocity = observation[1]
    if velocity > 0:
        action = 2
    else:
        action = 0
    return action


def get_index_of_max(values):
    # Ignore the max, the second variable returned
    index, _ = max(enumerate(values), key=operator.itemgetter(1))
    return index


class DQN:
    """A Deep Q-network.

    Novel elements:
    1. Deep neural network for learning the action-value: It is hard to converge
    when using function approximation, off-policy learning, and bootstrapping.
    2. Experience replay: We update from random experiences.
    3. Fixed Q targets: We hold the Q target constant for n steps.
    """

    def __init__(self, discount_rate, initial_epsilon, final_epsilon, epsilon_decay_steps, updates_per_freeze, policy_log_frequency, experiences_per_train, experience_cut_threshold, experiences_to_keep, render_frequency):
        self.env = gym.make("MountainCar-v0")
        # Future rewards for 1 step in the future are valued at
        # discount_rate * reward.
        assert (0 < discount_rate <= 1)
        self.discount_rate = discount_rate

        # We take a random action with probability epsilon.
        # Because we initialize our action-values optimistically (higher than
        # they will be after a few updates), we should(?) not need an epsilon > 0.
        assert (0 <= initial_epsilon <= 1)
        assert (0 <= final_epsilon <= 1)
        assert (0 < epsilon_decay_steps)
        self.epsilon = initial_epsilon
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.total_steps_ever_taken = 0

        self.experience_cut_threshold = experience_cut_threshold
        self.experiences_to_keep = experiences_to_keep

        # Update the frozen model after updates_per_freeze steps.
        # updates_per_freeze can be > max_steps for the episode.
        self.updates_per_freeze = updates_per_freeze
        self.model = init_model()
        self.frozen_model = None
        self.freeze_model()
        self.log_policy()

        self.policy_log_frequency = policy_log_frequency
        self.render_frequency = render_frequency
        self.goal_position = 0.5
        self.experiences = []
        self.experiences_per_train = experiences_per_train
        self.max_steps = 200
        self.sum_steps = 0
        self.num_episodes = 0
        self.updates_applied = 0

    def normalize(self, observation):
        # normalize using the env's observation space into the range of [-1, 1]^n
        low = self.env.observation_space.low
        high = self.env.observation_space.high
        result = [(2.0 * (observation[i] - low[i])) / (high[i] - low[i]) - 1 for i in range(len(low))]
        # TODO: Remove this assert once the code has run for a while
        for i in range(len(result)):
            assert (-1.0 <= result[i] <= 1.0)
        return result

    def get_action_values(self, model, observation):
        # Reshape the observation to be num_samples X num_inputs
        normalized = self.normalize(observation)
        reshaped = np.array(normalized).reshape(1, 2)
        return model.predict(reshaped)[0]

    def get_action(self, observation):
        """Take a random action with probability epsilon, and the action that
        our model predicts has the highest action-value otherwise.
        """
        rand_number = random.random()
        if rand_number < self.epsilon:
            return self.env.action_space.sample()
        else:
            action_values = self.get_action_values(self.model, observation)
            # Note, neural network output values correspond to action indices.
            # TODO: revisit to consider fewer forward passes.
            return get_index_of_max(action_values)

    def log_policy(self):
        logging.debug('Left-Right is position, higher is greater forward velocity, lower is greater backward velocity')
        logging.debug('Displaying the current model\'s policy:')
        self.log_policy_for_model(self.model)
        logging.debug('Displaying the frozen model\'s policy:')
        self.log_policy_for_model(self.frozen_model)

    def log_policy_for_model(self, model):
        """Display the policy for debugging purposes."""
        should_be_goodish = self.get_action_values(model, [0.4, 0.04])
        logging.debug('goodish: {}'.format(should_be_goodish))
        should_be_badish = self.get_action_values(model, [-0.52359, 0.0])
        logging.debug('badish: {}'.format(should_be_badish))
        for velocity in reversed(range(-7, 8)):
            row_string = ''
            for position in range(-12, 7):
                test_observation = [position * 0.1, velocity * 0.01]
                prediction = self.get_action_values(model, test_observation)
                action = get_index_of_max(prediction)
                # action = human_policy(test_observation)
                row_string += str(action)
            logging.debug(row_string)

    def decay_epsilon(self):
        if self.total_steps_ever_taken > self.epsilon_decay_steps:
            self.epsilon = self.final_epsilon
            return
        # we know that self.total_steps_ever_taken < self.epsilon_decay_steps
        decay_per_step = (self.initial_epsilon - self.final_epsilon) / self.epsilon_decay_steps
        self.epsilon = self.initial_epsilon - (self.total_steps_ever_taken * decay_per_step)

    def sample_episode(self):
        """Take an action given the policy, store the experience,
        and update the model.
        """
        observation = self.env.reset()
        for step in range(1, self.max_steps + 1):
            self.total_steps_ever_taken += 1
            if self.num_episodes % self.render_frequency == 0:
                self.env.render()

            # The model learns faster if given a little help :)
            # if self.num_episodes % 5 == 0:
            #     action = human_policy(observation)
            # else:
            action = self.get_action(observation)
            # The last variable, info, is always {}.
            # The second to last variable, done, is degenerate.
            initial_observation = observation
            observation, reward, _, _ = self.env.step(action)
            position = observation[0]
            found_goal = bool(position >= self.goal_position)
            self.decay_epsilon()
            # logging.debug('action: {}, obs: {}, rew: {}, found_goal: {}'
            #              .format(action, observation, reward, found_goal))

            # Store each experience.
            experience = Experience(initial_observation,
                                    action,
                                    reward,
                                    observation,
                                    found_goal)
            self.experiences.append(experience)

            if not found_goal and step < self.max_steps:
                self.train_model(verbose=False)
                self.updates_applied += 1
                if self.updates_applied == self.updates_per_freeze:
                    self.updates_applied = 0
                    self.freeze_model()

            # Terminate the episode if we've found the goal
            else:
                # Print out the loss at the end of every episode.
                self.train_model(verbose=True)

                # Log the average number of steps to complete the episode.
                self.sum_steps += step
                self.num_episodes += 1
                avg_steps = self.sum_steps / self.num_episodes
                logging.debug('Episode {} complete in {} steps. New step average {}'
                              .format(self.num_episodes, step, avg_steps))
                logging.debug('Epsilon is {}'.format(self.epsilon))
                break

        num_experiences = len(self.experiences)
        if num_experiences > self.experience_cut_threshold:
            self.experiences = self.experiences[num_experiences - self.experiences_to_keep:]
        if self.num_episodes % self.policy_log_frequency == 0:
            self.log_policy()

    def train_model(self, verbose=False):
        """Train on samples from the experience buffer."""
        if self.num_episodes < 1:
            return

        # Sample items from the experience buffer at random.
        batch_of_experiences = random.sample(self.experiences, self.experiences_per_train)
        # batch_of_experiences = [self.experiences[-1]]
        mini_batch_xs = list(map(lambda x: self.normalize(x.old_observation), batch_of_experiences))

        # For each experience, calculate the target based on the frozen model.
        mini_batch_ys = list(map(lambda i_el: self.calculate_targets(i_el[1],
                                                                     verbose=(verbose and i_el[0] == 0)),
                                 enumerate(batch_of_experiences)))
        xs = np.array(mini_batch_xs)
        ys = np.array(mini_batch_ys)

        # Occasionally log model training results.
        self.model.fit(xs, ys, verbose=int(verbose))

    # TODO: Debug me, probably
    def calculate_targets(self, experience, verbose=False):
        """Make an update to the action-value based on the reward from one step."""
        # TODO: what should q_action_values be named, really?
        q_action_values = self.get_action_values(self.model, experience.old_observation)
        # If we found the goal, the target action value is the reward.
        target = experience.reward
        if not experience.found_goal:
            q_hat_action_values = self.get_action_values(self.frozen_model, experience.new_observation)
            future_reward = max(q_hat_action_values)
            discounted_future_reward = self.discount_rate * future_reward
            target += discounted_future_reward
            if verbose:
                logging.debug('q: {}, q_hat: {}, target: {}, old_obs: {}, new_obs: {}'
                              .format(q_action_values, q_hat_action_values, target,
                                      experience.old_observation, experience.new_observation))
        # Set the action-value of the action that was taken to the target.
        # The other two action-values are not changed, resulting in 0 error for those actions.
        q_action_values[experience.action] = target
        return q_action_values

    def freeze_model(self):
        """Make a copy of the model."""
        copy_of_model = tf.keras.models.clone_model(self.model)
        copy_of_model.set_weights(self.model.get_weights())
        self.frozen_model = copy_of_model


def main():
    dqn = DQN(
        discount_rate=1.0,
        initial_epsilon=1.0,
        final_epsilon=0.1,
        epsilon_decay_steps=10000,
        updates_per_freeze=500,
        policy_log_frequency=50,
        experiences_per_train=32,
        experience_cut_threshold=11000,
        experiences_to_keep=10000,
        render_frequency=25
    )
    # TODO: make a param.
    while True:
        dqn.sample_episode()


if __name__ == '__main__':
    # TODO: argparse
    main()
