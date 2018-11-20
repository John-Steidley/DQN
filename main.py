from collections import namedtuple
import gym
import tensorflow as tf
import random
import operator
import numpy as np
# env produces zero acceleration at pi/6 ~= 0.52359

Experience = namedtuple('Experience', ['old_observation', 'action', 'reward', 'new_observation'])

def init_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(3, input_shape=(2,))
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
    def __init__(self):
        self.env = gym.make("MountainCar-v0") 
        self.model = init_model()
        self.experiences = []
        self.sum_steps_per_episode = 0
        self.num_episodes = 0

    def get_epsilon(self):
        # TODO() decay epsilon
        return 0.05

    def get_action(self, observation):
        epsilon = self.get_epsilon()
        rand_number = random.random()
        # Take a random action if less than epsilon
        # Note, neural network output values corrrespond to action indices
        if rand_number < epsilon:
            print('random action')
            return self.env.action_space.sample()        
        else:
            action_values = self.model.predict(np.array(observation).reshape(1,2))[0]
            print('action values', action_values, 'for', observation)
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
            print('action is', action)
            # The last variable, info, is always {}
            # The second to last variable, done, is degenerate
            initial_observation = observation
            observation, reward, _, _ = self.env.step(action)
            # Store each experience
            experience = Experience(initial_observation, action, reward, observation)
            self.experiences.append(experience)
            position = observation[0]
            # 0.5 is the goal position
            done = bool(position >= 0.5) or i == max_steps
            print(action, observation, reward, done)
            
            if done:
                self.sum_steps_per_episode += i
                self.num_episodes += 1

        avg_steps =  self.sum_steps_per_episode / float(self.num_episodes)
        print('Done!', avg_steps)


def main():
    dqn = DQN()
    for i in range(1):
        dqn.sample_episode()


if __name__ == '__main__':
    main()