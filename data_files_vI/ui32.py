import torch.nn as nn  # all neural network modules
import torch.nn.functional as functional


class NN (nn.Module): # inherits nn.Module (NN = Neural Network)
    def __init__(self, input_size, num_classes): # for mnist it is 28*28 = 784
        super(NN, self).__init__()

        # self.hidden_layer_number = 50
        # self.fc1 = nn.Linear(input_size, self.hidden_layer_number) # 50 - random number for the hidden layer
        # self.fc2 = nn.Linear(self.hidden_layer_number, num_classes)

        # two hidden layers
        self.fc1 = nn.Linear(input_size, 250)
        self.fc2 = nn.Linear(250, 60)
        self.fc3 = nn.Linear(60, num_classes)

    def forward(self, tensor):
        # tensor = functional.relu(self.fc1(tensor))
        # tensor = self.fc2(tensor)
        # return tensor
        # return self.fc2(functional.relu(self.fc1(tensor)))
        return self.fc3(functional.tanh(self.fc2(functional.tanh(self.fc1(tensor)))))
