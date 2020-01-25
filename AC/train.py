import gym, getopt, sys
from ac import ACAgent as Agent
#from ac_suboptimal import ACSuboptimalAgent as Agent
import numpy as np
from monitor import Monitor
from singletrainer import SingleTrainer
from smoothie import Smoothie
from envs import make_env

opts, args = getopt.getopt(sys.argv[1:], "r:t:vg:T:c:")
env_name = args[0]

opts = dict(opts)
report_interval = int(opts.get("-r", 1))
test_interval = int(opts.get("-t", 100))
render = "-v" in opts
gamma = float(opts.get("-g", 0.99))
comment = opts.get("-c", "")

env = make_env(env_name)
num_actions = env.action_space.n
observation_shape = env.observation_space.shape
assert len(observation_shape) == 1
observation_dim = observation_shape[0]

agent = Agent(observation_dim, num_actions, 0.00001, 0.00005, gamma=gamma)

title = opts.get("-T","Training agent %s in %s" % (agent.__class__.__name__, env_name))


score_history = []
num_episodes = 10000
monitor = Monitor("monitor.csv",
    title = title,
    metadata = dict(
        gamma=0.99,
        agent_class=Agent.__name__,
        pretrain=0,
        copies=1,
        report_interval=1,
        comment = comment,
        environment = env.__class__.__name__
    ),
    plots=[
        [
            {
                "label":        "min test score",
                "line_width":   1.0
            },
            {
                "label":        "average test score",
            },
            {
                "label":        "max test score",
                "line_width":   1.0
            }
        ],
        [
            {
                "label":        "min train score",
                "line_width":   1.0
            },
            {
                "label":        "train score"
            },
            {
                "label":        "max train score",
                "line_width":   1.0
            }
        ]
]
)
monitor.start_server(8080)
trainer = SingleTrainer(agent, env)
score_smoother = Smoothie(0.01)

next_test = test_interval

for t, score in trainer.train(10000, report_interval=report_interval):
    min_score, score_ma, max_score = score_smoother.update(score)
    print("Training: episodes=%4d score: %.3f" % (t, score))
    monitor.add(t, {
        "train score":      score, 
        "min train score":  min_score, 
        "max train score":  max_score
    })
    if t >= next_test:
        min_test, mean_test, max_test = trainer.test(10, render=render)
        print("Test after %d train episodes:" % (t,), "    min, mean, max score:", min_test, mean_test, max_test)
        monitor.add(t,
            {
                "min test score":   min_test,
                "max test score":   max_test,
                "average test score":   mean_test
            }
        )
        next_test += test_interval
        
    
