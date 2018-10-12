from keras import layers, models, optimizers
from keras import backend as K


def build_network(base_size, num_layers, state_size, action_size):
    states = layers.Input((state_size,), name="states")
    network = states

    for i in range(num_layers):
        network = layers.Dense(base_size * 2 ** i, activation="relu")(network)
        network = layers.BatchNormalization()(network)
        network = layers.Dropout(0.2)(network)
    actions = layers.Dense(action_size, activation='softmax', name="actions")(network)

    return states, actions


class ActorNetwork:
    """The actor network also known as a policy network is a map from states to actions: Given a state, it returns a softmax output over the action space. Learning is through back propagation of the action gradients estimated in the critic network """

    def __init__(self, state_size, action_size, base_size=64, num_layers=3, learning_rate=0.00001):
        self.state_size = state_size
        self.action_size = action_size
        self.base_size = base_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        states, actions = build_network(self.base_size, self.num_layers,
                                        self.state_size, self.action_size)

        self.model = models.Model(inputs=states, outputs=actions)
        action_grads = layers.Input(
            shape=(self.action_size,))  # Input layer that takes in action gradients from the critic network
        loss = K.mean(-action_grads * actions)  # Loss function, whose gradients maximize the expected rewards
        # The negative sign is used in order for the actor to maximize the expected q values as the optimization process minimizes loss
        train_updates = optimizers.Adam(lr=self.learning_rate).get_updates(params=self.model.trainable_weights,
                                                                           loss=loss)
        self.custom_train = K.function(
            inputs=[self.model.input, action_grads],
            outputs=[],
            updates=train_updates)  # Custom Training function for the actor network that makes use of the Q gradients w.r.t the action probabilities. With this custom function, the training aims at maximing i.e minimizing the negative of the q values
