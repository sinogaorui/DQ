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
            env.env.render()
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
            t11 = threading.Thread(target=make_Action, args=(agent,state,10, )) 
            t12 = threading.Thread(target=make_Action, args=(agent,state,11,))
            t13 = threading.Thread(target=make_Action, args=(agent,state,12, )) 
            t14 = threading.Thread(target=make_Action, args=(agent,state,13,))
            t15 = threading.Thread(target=make_Action, args=(agent,state,14, )) 
            t16 = threading.Thread(target=make_Action, args=(agent,state,15,))
            t17 = threading.Thread(target=make_Action, args=(agent,state,16, )) 
            t18 = threading.Thread(target=make_Action, args=(agent,state,17,))
            t19 = threading.Thread(target=make_Action, args=(agent,state,18, )) 
            t20 = threading.Thread(target=make_Action, args=(agent,state,19,))
            t21 = threading.Thread(target=make_Action, args=(agent,state,20, )) 
            t22 = threading.Thread(target=make_Action, args=(agent,state,21,))
            t23 = threading.Thread(target=make_Action, args=(agent,state,22, )) 
            t24 = threading.Thread(target=make_Action, args=(agent,state,23,))
            t25 = threading.Thread(target=make_Action, args=(agent,state,24, )) 
            t26 = threading.Thread(target=make_Action, args=(agent,state,25,))
            t27 = threading.Thread(target=make_Action, args=(agent,state,26, )) 
            t28 = threading.Thread(target=make_Action, args=(agent,state,27,))
            t29 = threading.Thread(target=make_Action, args=(agent,state,28, )) 
            t30 = threading.Thread(target=make_Action, args=(agent,state,29,))
            t31 = threading.Thread(target=make_Action, args=(agent,state,30, )) 
            t32 = threading.Thread(target=make_Action, args=(agent,state,31,))
            t33 = threading.Thread(target=make_Action, args=(agent,state,32, )) 
            t34 = threading.Thread(target=make_Action, args=(agent,state,33,))
            t35 = threading.Thread(target=make_Action, args=(agent,state,34, )) 
            t36 = threading.Thread(target=make_Action, args=(agent,state,35,))
            t37 = threading.Thread(target=make_Action, args=(agent,state,36, )) 
            t38 = threading.Thread(target=make_Action, args=(agent,state,37,))
            t39 = threading.Thread(target=make_Action, args=(agent,state,38, )) 
            t40 = threading.Thread(target=make_Action, args=(agent,state,39,))
			
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
            t11.start() 
            t12.start() 
            t13.start() 
            t14.start() 
            t15.start() 
            t16.start() 
            t17.start() 
            t18.start()  
            t19.start() 
            t20.start()  
            t21.start() 
            t22.start() 
            t23.start() 
            t24.start() 
            t25.start() 
            t26.start() 
            t27.start() 
            t28.start()  
            t29.start() 
            t30.start() 
            t31.start() 
            t32.start() 
            t33.start() 
            t34.start() 
            t35.start() 
            t36.start() 
            t37.start() 
            t38.start()  
            t39.start() 
            t40.start() 
			
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
            t11.join() 
            t12.join() 
            t13.join() 
            t14.join() 
            t15.join() 
            t16.join() 
            t17.join() 
            t18.join()  
            t19.join() 
            t20.join()  
            t21.join() 
            t22.join() 
            t23.join() 
            t24.join() 
            t25.join() 
            t26.join() 
            t27.join() 
            t28.join()  
            t29.join() 
            t30.join() 	
            t31.join() 
            t32.join() 
            t33.join() 
            t34.join() 
            t35.join() 
            t36.join() 
            t37.join() 
            t38.join()  
            t39.join() 
            t40.join() 				
			
         
			
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