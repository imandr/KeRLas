import gym
from ac import Agent
import numpy as np
from monitor import Monitor
from trainer import Trainer
from smoothie import Smoothie

agent = Agent(0.00001, 0.00005)
env = gym.make("LunarLander-v2")
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
        comment = "Single agent training",
        environment = env.__class__.__name__
    ),
    plots=[
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
trainer = Trainer(agent, env, train_interval=1, mb_size=30, shuffle=False, reverse=False)
score_smoother = Smoothie(0.01)

for t in range(num_episodes):
    score, _, actor_metrics, critic_metrics = trainer.run_episode(learn=True)
    min_score, score_ma, max_score = score_smoother.update(score)
    print(t, score, min_score, score_ma, max_score, actor_metrics, critic_metrics)
    monitor.add(t, {
        "min train score":  min_score,
        "train score":      score_ma, 
        "max train score":  max_score
    })
