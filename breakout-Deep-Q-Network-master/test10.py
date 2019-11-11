"""

### NOTICE ###
You DO NOT need to upload this file

"""

import argparse
import numpy as np
from environment import Environment
import datetime
import timeit
import threading 

seed = 11037

def parse():
    parser = argparse.ArgumentParser(description="MLDS 2018 HW4")
    parser.add_argument('--test_pg', action='store_true', help='whether test policy gradient')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args

def make_Action(agent, state, i):
    #print("thread = ", i)
    return agent.make_action(state, test=True)

def test(agent, env, total_episodes=30):
    rewards = []
    env.seed(seed)
    #count = 0
    #totalTime100 = 0	
    for i in range(total_episodes):
        state = env.reset()
        agent.init_game_setting()
        done = False
        episode_reward = 0.0
        count = 0
        totalTime100 = 0
		
        #playing one game
        while(not done):
            count = count + 1
            #env.env.render()
            action = None
            #print("Start time = ",datetime.datetime.now().time() )
            start = timeit.default_timer() #a= datetime.datetime.now().time()
            
            t1 = threading.Thread(target=make_Action, args=(agent,state,0, )) 
            t2 = threading.Thread(target=make_Action, args=(agent,state,1,))
            t3 = threading.Thread(target=make_Action, args=(agent,state,2, )) 
            t4 = threading.Thread(target=make_Action, args=(agent,state,3,))
            t5 = threading.Thread(target=make_Action, args=(agent,state,4, )) 
            t6 = threading.Thread(target=make_Action, args=(agent,state,5,))
            t7 = threading.Thread(target=make_Action, args=(agent,state,6, )) 
            t8 = threading.Thread(target=make_Action, args=(agent,state,7,))
            t9 = threading.Thread(target=make_Action, args=(agent,state,8, )) 
            t10 = threading.Thread(target=make_Action, args=(agent,state,9,))

            t1.start() 
            t2.start() 
            t3.start() 
            t4.start() 
            t5.start() 
            t6.start() 
            t7.start() 
            t8.start()  
            t9.start()
            t10.start() 

            t1.join() 
            t2.join() 
            t3.join() 
            t4.join() 
            t5.join() 
            t6.join() 
            t7.join() 
            t8.join()  
            t9.join()
            t10.join() 
		
            # all threads completely executed        
            end = timeit.default_timer()# b= datetime.datetime.now().time()
            total = end-start
            totalTime100 = totalTime100 + total
            print("Done!")
			
            #print("TotalTime = ", total, " , aver = ", total/100 )	
            action = agent.make_action(state, test=True)
            state, reward, done, info = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
        x= totalTime100/count
        print("Totaltime100= ",x)# " , time for 1 agent = ", x/100)
        print('[ episode ', i, '] upclipped reward :', episode_reward)
    print('Run %d episodes'%(total_episodes))
    print('Mean:', np.mean(rewards))
    #x= totalTime100/count
    #print("Totaltime100= ",x)# " , time for 1 agent = ", x/100)


def run(args):
    if args.test_pg:
        env = Environment('Pong-v0', args, test=True)
        from agent_dir.agent_pg import Agent_PG
        agent = Agent_PG(env, args)
        test(agent, env)

    if args.test_dqn:
        env = Environment('BreakoutNoFrameskip-v4', args, atari_wrapper=True, test=True)
        from agent_dir.agent_dqn import Agent_DQN
        agent = Agent_DQN(env, args)
        test(agent, env, total_episodes=5)#100


if __name__ == '__main__':
    args = parse()
    run(args)
