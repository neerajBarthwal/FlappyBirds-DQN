from DQNModel import DQNModel
from QLearn import Qlearn
from docutils.nodes import target
if __name__ == '__main__':
    dqn1 = DQNModel(0.01)
    dqn2 = DQNModel(0.01)
    active_network = dqn1.model
    target_network = dqn2.model
    game = Qlearn(active_network,target_network)
    game.play_game() 