from DQNModel import DQNModel
from QLearn import Qlearn
from keras.models import model_from_json
if __name__ == '__main__':
    
#     if 1==2:
#         # load json and create model
#         json_file = open('model.json', 'r')
#         loaded_model_json = json_file.read()
#         json_file.close()
#         loaded_model = model_from_json(loaded_model_json)
#         # load weights into new model
#         loaded_model.load_weights("active_model.h5")
#         print("Loaded model from disk")
    
    dqn1 = DQNModel(learning_rate=1e-6)
    dqn2 = DQNModel(learning_rate=1e-6)
    active_network = dqn1.model
    target_network = dqn2.model
    game = Qlearn(active_network,target_network)
    game.play_game() 