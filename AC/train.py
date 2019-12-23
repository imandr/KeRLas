from simple_env import SimpleEnv
from cartpole import TimedCartPoleEnv
from lunar_lander import LunarLander
from model import QVModel
from kerlas import ReplayMemory
from kerlas.policies import BoltzmannQPolicy, GreedyEpsPolicy
import numpy as np, random
from time_limit import TimeLimit

np.set_printoptions(precision=4, suppress=True)

Gamma = 0.99
#game_env = SimpleEnv(200)


# lunar lander
env = LunarLander()
game_env = TimeLimit(env, time_limit=300, timeout_reward=-1.0)

state_dim = game_env.observation_space.shape[-1]
nactions = game_env.action_space.n

print("state_dim:", state_dim)

qvmodel = QVModel(state_dim, nactions, Gamma)
memory = ReplayMemory(10000)

NGames = 100000
NextTrain = TrainInterval = 5     # train after 5 games
NextTest = TestInterval = 500

def play_game(env, model, policy, log=False, render=False):
    x = env.reset()
    if render:
        env.render()
    done = False
    score = 0.0
    record = []
    t = 0
    if log:
        print("----------------------------")
    while not done:
        q = model.q(x)
        a = policy(q)
        x1, r, f, info = env.step(a)
        if render:
            env.render()
        record.append((x, a, r, x1, 1.0 if f else 0.0, info))
        score += r
        if log:
            v = model.v(x)
            print(t, x, v, q, a, r, x1, f, info)
        x = x1
        done = f
        t += 1
    if log:
        print("Game over. Score=%.5f" % (score,))
    return score, record
    
def calc_v_estimate(model, record, gamma):
    n_ahead = 3
    x, a, r, x1, f, info = zip(record)
    f = np.array(f)
    x = np.array(x)
    r = np.array(r)
    x1 = np.array(x1)
    v1 = model.v_array(x1) * (1-f)
    
    
    
def test(env, qmodel, policy, render=False):
    ngames = 100
    sum_scores = 0.0
    for _ in range(ngames):
        score, _ = play_game(env, qmodel, policy)
        sum_scores += score
    print("Average score:", sum_scores/ngames)
    play_game(env, qmodel, policy, log=True, render=render)
    
def fit(memory, verbose):
    batch_size = 1000
    #random.shuffle(memory)
    n = len(memory)
    j = 0
    for ib in range(0, n, batch_size):
        batch = memory[ib:ib+batch_size]
        x0, a, r, x1, f, info = zip(*batch)
        a = np.array(a)
        x0 = np.array(x0)
        x1 = np.array(x1)
        r = np.array(r)
        f = np.array(f)
        v0 = qvmodel.v_array(x0)
        v1 = qvmodel.v_array(x1) * (1.0-f)
        error = v1*Gamma + r - v0
        metrics = qvmodel.train(x0, v0, a, r, x1, v1, f, verbose and ib == 0)
        if verbose and ib == 0:
            print(metrics)
    
test_policy = BoltzmannQPolicy(0.00001)     
memory = []

train_policies = [GreedyEpsPolicy(0.3), BoltzmannQPolicy(0.01), GreedyEpsPolicy(0.01), BoltzmannQPolicy(0.001), GreedyEpsPolicy(0.1), BoltzmannQPolicy(0.0001)]
ip = 0


for igame in range(NGames):

    tp = train_policies[ip]
    ip = (ip+1)%len(train_policies)
    score, record = play_game(game_env, qvmodel, tp)
    memory += record
    for x0, a, r, x1, f, info in record:
        if r > 0.75:
            print ("big reward:", x0, a, r, x1, f, info)
    
    if igame >= NextTrain:
        verbose = igame and igame % 100 == 0
        fit(memory, verbose)
        memory = []
        NextTrain += TrainInterval
    
    if igame >= NextTest:
        test(game_env, qvmodel, test_policy, render=True)
        NextTest = igame + TestInterval
        

    
