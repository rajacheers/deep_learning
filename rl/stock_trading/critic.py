from keras import layers, models, optimizers
from keras import backend as K


def build_network(base_size, num_layers, state_size, action_size):
    states = layers.Input((state_size,), name="states")
    actions = layers.Input((action_size,), name="actions")
    network = layers.concatenate([actions, states], axis=1)

    for i in range(num_layers):
        network = layers.Dense(base_size * 2 ** i, activation="relu")(network)
        network = layers.BatchNormalization()(network)
        network = layers.Dropout(0.2)(network)
    Q_value = layers.Dense(units=1)(network)
    return states, actions, Q_value


class CriticNetwork:

    def __init__(self, state_size=10, action_size=3, base_size=32, num_layers=2, learning_rate=0.0000001):
        """The critic network serves as a discriminator or value model that returns the Q values of selected actions
        State size refers to the number of previous days(window) stock data to consider in the model
        The action size on the other hand reders to the number of possible actions(3 in this case): Buy, sell, sit. The critic model takes has two different inputs: the states and the actions, then returns the value of the pair  """
        self.state_size = state_size
        self.action_size = action_size
        self.base_size = base_size
        self.num_layers = num_layers
        states, actions, Q_values = build_network(self.base_size, self.num_layers,
                                                  self.state_size, self.action_size)
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Train with Adam optimizer on mean squared error loss
        self.model.compile(optimizer=optimizers.Adam(learning_rate), loss='mse')

        # The action gradients represent the derrivative of the Q values with respect to the the action and are an essential part of the actor critic architecture as they are backpropagated into the actor network
        action_gradients = K.gradients(Q_values, actions)

        """This is a custom function that maps a given state action pair to an action gradient. This cutsom function enables the actor network to fectch gradients for any given action based on the critic network outputs
        the input is the state action pair and the output is the gradient of actions with respect to the action logits"""
        self.get_action_gradients = K.function(self.model.input,
                                               action_gradients)  # returns the action gradients as a numpy array to be fed into the actor network