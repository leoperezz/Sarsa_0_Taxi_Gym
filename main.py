import gym
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

env = gym.make('Taxi-v3')
n_states=env.observation_space.n
n_actions=env.action_space.n
gamma=0.95
alpha=0.01 #learning rate
num_episodes=50000 

Q=np.zeros((n_states,n_actions)) #Q_values

def main():

    epsilon=0.3
    epsilon_decay=0.00005
    policy=np.zeros(shape=(n_states,1))
    sum_total=list()

    for i in range(num_episodes):

        reward_sum=0

        print(f'POLICY{i+1}: {policy}')

        state=env.reset()

        action=epsilon_policy(policy,epsilon,state)
        
        #epsilon converges
        if epsilon>0.01:
            epsilon-=epsilon_decay

        while 1:

            next_state,reward,done,prob=env.step(action)

            reward_sum+=reward

            action_f=epsilon_policy(policy,epsilon,next_state)

            if done:
                Q[state,action]=Q[state,action]+alpha*(reward-Q[state,action])
                break

            Q[state,action]=Q[state,action]+alpha*(reward+gamma*Q[next_state,action_f]-Q[state,action])

            policy[state]=np.argmax(Q[state,:])

            state=next_state
            action=epsilon_policy(policy,epsilon,next_state)

        sum_total.append(reward_sum)       
    
    sum_total=np.array(sum_total)
    plt.plot(sum_total)
    plt.show()
    play_game(policy)
    print(Q)

    return 0;


def epsilon_policy(policy,value,state):

    r=np.random.random_sample()

    if(r<value):
        return np.random.randint(n_actions)
    else:
        return policy[state]  


def play_game(policy):

    state=env.reset()
    action=policy[state]
    max_play=0

    while(max_play<50):

        next_state,reward,done,_=env.step(action)

        env.render()
        time.sleep(1)

        if done:
            break

        action=policy[next_state]

        max_play+=1




if __name__ == '__main__':
    main()
