from env import Env
from model import QVModel, train_policy, test_policy
from memory import ReplayMemory
import numpy as np


Gamma = 0.9
game_env = Env(50)
qvmodel = QVModel(game_env.StateDim, game_env.NActions, Gamma)
memory = ReplayMemory(10000)

NGames = 100000
NextTrain = TrainInterval = 100     # train after 100 games
NextTest = TestInterval = 500

def play_game(env, model, policy, log=False):
    x = env.reset()
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
        record.append((x, a, r, x1, 1.0 if f else 0.0))
        score += r
        if log:
            print(t, x, q, a, r, x1, f)
        x = x1
        done = f
    return score, record
    
def train(memory, qvmodel):
    mbsize = 30
    nbatches = 100
    for b in range(nbatches):
        sample = memory.sample(mbsize)
        x0, a0, r, v0, x1, v1, f = zip(*sample)
        v_metrics = qvmodel.train_v(np.array(x0), np.array(v0), np.array(r), np.array(x1), np.array(f))
    for b in range(nbatches):
        sample = memory.sample(mbsize)
        x0, a0, r, v0, x1, v1, f = zip(*sample)
        q_metrics = qvmodel.train_q(np.array(x0), np.array(a0), np.array(r), np.array(x1))
    print("Traning metrics: q: %.6f  v: %.6f" % (q_metrics, v_metrics))
        
def test(env, qmodel, policy):
    ngames = 100
    sum_scores = 0.0
    for _ in range(ngames):
        score, _ = play_game(env, qmodel, policy)
        sum_scores += score
    print("Average score:", sum_scores/ngames)
    play_game(env, qmodel, policy, log=True)

for igame in range(NGames):
    score, record = play_game(game_env, qvmodel, train_policy)
    
    x0, a, r, x1, f = zip(*record)
    a = np.array(a)
    x0 = np.array(x0)
    x1 = np.array(x1)
    r = np.array(r)
    f = np.array(f)
    v0 = qvmodel.v_array(x0)
    v1 = qvmodel.v_array(x1) * (1.0-f)
    error = v1*Gamma + r - v0
    #print("v0:", v0)
    #print("v0p:", v1*Gamma + r)
    qvmodel.train_v(x0, v1*Gamma + r)
    qvmodel.train_q(x0, a, error)
    
    if igame >= NextTest:
        test(game_env, qvmodel, test_policy)
        NextTest = igame + TestInterval
        

    
