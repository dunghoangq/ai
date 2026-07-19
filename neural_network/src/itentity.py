from .activation import Activation

class Identity(Activation):

    def forward(self, x):
        return x