import pickle
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

available_actions = load_obj('available_actions')
ast = load_obj("ast")

for action in available_actions:
    new_query = available_actions[action]['method'](ast, available_actions[action])
    print(new_query)


