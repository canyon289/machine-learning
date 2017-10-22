import gym
import numpy as np
import ipdb

env = gym.make('FrozenLake-v0')

# Make a Q learning table that has the spaces and actions
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Set Learning parameters
alpha = .8
gamma = .95

num_episodes = 2000

# Lists for total rewards and steps per episode
reward_list = []

for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    d = False
    j=0
    
    while j<99:
        env.render()
        j+=1
        # Choose an action by greedily picking from Q Table
        """
        I believe this means its an on policy learner as it uses the optimal
        action from the observation space while learning
        """
        # What is happening here is what I need to figure out
        act_l = Q[state,:] + np.random.randn(1, env.action_space.n)*(1./(i+1))
        act =np.argmax(act_l)
        s1, r, d,_ = env.step(act)

        # Update Q Table with new knowledge
        #What is S1? Is it next step
        Q[state,act] = Q[state,act] + alpha*(r+gamma*np.max(Q[s1,:]) - Q[state,act])
        rAll += r
        s=s1
        if d == True:
            break

    reward_list.append(rAll)
    print("Next Simulation")


print("Score over time: " +  str(sum(reward_list)/num_episodes))
print(Q)
