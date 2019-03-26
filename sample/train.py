from KeRLas import Brain, defaultQModel
from env import Env
from KeRLas.models import DirectDiffModel
from KeRLas.policies import BoltzmannQPolicy


env = Env()
rlmodel = DirectDiffModel(
    defaultQModel(env.StateDim, env.NActions), 
    0.9)
policy = BoltzmannQPolicy(0.01)
brain = Brain(rlmodel, policy, 10000, 0.5)

mbsize = 20

brain.training_model().fit_generator(
    brain.trainig_data_generator(mbsize),
    steps_per_epoch = 1000, epochs = 100, verbose = True
)


