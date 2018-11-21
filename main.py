from collections import namedtuple
import gym
import tensorflow as tf
import random
import operator
import numpy as np
# env produces zero acceleration at pi/6 ~= 0.52359

Experience = namedtuple('Experience', [
    'old_observation',
    'action',
    'reward',
    'new_observation',
    'done'
])

def init_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(8, activation = 'relu', input_shape=(2,)),
        tf.keras.layers.Dense(3)
    ])
    # TODO: Consider switching to RMSProp to match the paper
    model.compile(optimizer='SGD', loss='mean_squared_error')
    return model

# probably not quite optimal, but quite good
# achieves ~ 129.9 avg reward
# 2 is accelerate forward
# 1 is do nothing
# 0 is accelerate backwards
def human_policy(observation):
    velocity = observation[1]
    if velocity > 0:
        action = 2
    else:
        action = 0
    return action

def get_index_of_max(values):
    # We don't value, the second value returned
    index, _ = max(enumerate(values), key=operator.itemgetter(1))
    return index

class DQN:
    def __init__(self, gamma, updates_per_freeze):
        self.gamma = gamma
        self.env = gym.make("MountainCar-v0") 
        self.model = init_model()
        self.frozen_model = None
        self.freeze_model()
        self.experiences = []
        self.sum_steps_per_episode = 0
        self.num_episodes = 0
        self.updates_applied = 0
        self.updates_per_freeze = updates_per_freeze

    def get_epsilon(self):
        # TODO() decay epsilon
        return 0.2

    def get_action_values(self, model, observation):
        return model.predict(np.array(observation).reshape(1,2))[0]

    def get_action(self, observation):
        epsilon = self.get_epsilon()
        rand_number = random.random()
        # Take a random action if less than epsilon
        # Note, neural network output values corrrespond to action indices
        if rand_number < epsilon:
            # print('random action')
            return self.env.action_space.sample()        
        else:
            action_values = self.get_action_values(self.model, observation)
            # print('action values', action_values, 'for', observation)
            # TODO() revisit to consider fewer forward passes
            return get_index_of_max(action_values)

    def sample_episode(self):
        observation = self.env.reset()
        # TODO() experiment with number of steps
        max_steps = 500
        for i in range(1, max_steps + 1):
            self.env.render()
            # action = human_policy(observation)
            action = self.get_action(observation)
            # print('action is', action)
            # The last variable, info, is always {}
            # The second to last variable, done, is degenerate
            initial_observation = observation
            observation, reward, _, _ = self.env.step(action)
            position = observation[0]
            # 0.5 is the goal position
            done = bool(position >= 0.5) or i == max_steps
            # print(action, observation, reward, done)
            # Store each experience
            experience = Experience(initial_observation, action, reward, observation, done)
            self.experiences.append(experience)
            self.update_model()
            if done:
                self.sum_steps_per_episode += i
                self.num_episodes += 1
                avg_steps =  self.sum_steps_per_episode / float(self.num_episodes)
                print('Done! This episode complete in', i, 'the new average is', avg_steps)
                return

    def update_model(self):
        if self.num_episodes < 1:
            return
        # Sample 32 items from the experience buffer at random
        batch_of_experiences = random.sample(self.experiences, 32)
        mini_batch_xs = list(map(lambda x: x.old_observation, batch_of_experiences))
        # evaluate using the frozen model
        # discount, and use the reward
        mini_batch_ys = list(map(self.transform_experience_into_training_point, batch_of_experiences))
        xs = np.array(mini_batch_xs)
        ys = np.array(mini_batch_ys)
        # print('xs', xs, 'with shape:', xs.shape)
        # print('ys', ys, 'with shape:', ys.shape)
        self.model.fit(xs, ys, verbose=0) # disable logging
        self.updates_applied += 1
        if self.updates_applied == self.updates_per_freeze:
            self.updates_applied = 0
            self.freeze_model()

    # TODO: Debug me, probabaly
    def transform_experience_into_training_point(self, experience):
        # WTF should this be named, really?
        q_action_values = self.get_action_values(self.model, experience.old_observation)
        target = experience.reward
        if not experience.done:
            q_hat_action_values = self.get_action_values(self.frozen_model, experience.new_observation)
            future_reward = max(q_hat_action_values)
            discounted_future_reward = self.gamma * future_reward
            target += discounted_future_reward
        # print('before', q_action_values, 'target:', target)
        q_action_values[experience.action] = target
        # print('after', q_action_values)
        return q_action_values

    def freeze_model(self):
        copy_of_model = tf.keras.models.clone_model(self.model)
        copy_of_model.set_weights(self.model.get_weights())
        self.frozen_model = copy_of_model
        # for debugging, display the policy
        for i in range(-24, 12):
            row_string = ''
            for j in range(-28, 32):
                test_observation = [i * 0.05, j * 0.0025]
                prediction = self.get_action_values(self.frozen_model, test_observation)
                action = get_index_of_max(prediction)
                # action = human_policy(test_observation)
                row_string += str(action)
            print(row_string)

def main():
    gamma = 0.99
    updates_per_freeze = 3000
    dqn = DQN(gamma, updates_per_freeze)
    for i in range(100):
        dqn.sample_episode()


if __name__ == '__main__':
    main()