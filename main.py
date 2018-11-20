import gym
import tensorflow as tf

# env produces zero acceleration at pi/6 ~= 0.52359

def init_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(3)
    ])
    # TODO: Consider switching to RMSProp to match the paper
    model.compile(optimzer='SGD', loss='mean_squared_error')
    return model

# probably not quite optimal, but quite good
# achieves ~ 129.9 avg reward
def human_policy(observation):
    velocity = observation[1]
    if velocity > 0:
        action = 2
    else:
        action = 0
    return action

def main():
    env = gym.make("MountainCar-v0")
    observation = env.reset()
    num_steps_in_episode = 0
    sum_steps_per_episode = 0
    num_successes = 0
    for _ in range(25000):
        # env.render()
        # 2 is accelerate forward
        # 1 is do nothing
        # 0 is accelerate backwards
        action = human_policy(observation)
        # The last variable, info, is always {}
        observation, reward, done, _ = env.step(action)
        num_steps_in_episode += 1
        # print(action, observation, reward, done)
        if done:
            # print(num_steps_in_episode)
            sum_steps_per_episode += num_steps_in_episode
            num_successes += 1
            num_steps_in_episode = 0
            observation = env.reset()
    avg_steps =  sum_steps_per_episode / float(num_successes)
    print('Done!', avg_steps)

if __name__ == '__main__':
    main()