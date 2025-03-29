import numpy as np
# manually aliasing np.bool8 to np.bool_, we silenced the warning while maintaining compatibility.
np.bool8 = np.bool_  # Force-alias the deprecated type

import gym
import random
import math

env = gym.make("Taxi-v3", render_mode="human")  

# change-able parameters:
discount_factor = 0.8 #gamma
delta_threshold = 0.00001 #is used to determine when to stop value iteration

# ε is used to define a stopping condition.
# If the difference between old and updated values is smaller than epsilon, the iteration stops.
# This prevents unnecessary computations once values stabilize.
epsilon = 1


def value_iteration(env, gamma, epsilon):
    # There are 500 discrete states since there are 25 taxi positions, 
    # 5 possible locations of the passenger (including the case when the 
    # passenger is in the taxi), and 4 destination locations.
    num_states = env.observation_space.n


    # There are 6 discrete deterministic actions:
    # 0: move south
    # 1: move north
    # 2: move east
    # 3: move west
    # 4: pickup passenger
    # 5: drop off passenger
    num_actions = env.action_space.n

    # Initialize the value function for each state
    V = np.zeros(num_states)

    #Write your code to implement value iteration main loop

        
        # taxi_row, taxi_col, passenger_location, destination = env.decode(41)
        # print(f"Taxi at ({taxi_row}, {taxi_col}), Passenger at {passenger_location}, Destination at {destination}")

    while True:

        delta = 0
        for state in range(num_states): #for each state
            v = V[state]

            action_decision = []
            for act in range(num_actions): # for every action

                # Taxi-v3 provides a model of the MDP in env.P, which is a dictionary 
                # describing all possible transitions from any state under each action.
                
                # env.P[state][action] is a list of (prob, next_state, reward, done)
                expected_return = 0
                for prob, next_state, reward, done in env.P[state][act]:
                    expected_return += prob * (reward + gamma * V[next_state])
                action_decision.append(expected_return)              

            max_value  =  max(action_decision) # best action value
            V[state] = max_value
            delta = max(delta, abs(v - V[state])) # Tracking the maximum difference
        
        if delta < epsilon: # Checking for convergence
            break

    # For each state, the policy will tell you the action to take
    policy = np.zeros(num_states, dtype=int)

    # Write your code here to extract the optimal policy from value function.
    for state in range(num_states): #for each state

        action_decision = []
        for act in range(num_actions): # for every action

            # Taxi-v3 provides a model of the MDP in env.P, which is a dictionary 
            # describing all possible transitions from any state under each action.
            
            # env.P[state][action] is a list of (prob, next_state, reward, done)
            expected_return = 0
            for prob, next_state, reward, done in env.P[state][act]:
                expected_return += prob * (reward + gamma * V[next_state])
            action_decision.append(expected_return)    
        policy[state] = np.argmax(action_decision) 


    return policy, V


# Run value iteration
policy, V = value_iteration(env, discount_factor, delta_threshold)


# resetting the environment and executing the policy
state = env.reset()
step = 0
done = False
state = state[0]

max_steps = 100
for step in range(max_steps):

    # Getting max value against that state, so that we choose that action
  
    action = policy[state]
    new_state, reward, done, truncated, info = env.step(action) # information after taking the action
    env.render()
    if done:
        print("number of steps taken:", step)
        break

    state = new_state
    
env.close()