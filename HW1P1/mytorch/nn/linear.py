import numpy as np


class Linear:

    def __init__(self, in_features, out_features, debug=False):
        """
        Initialize the weights and biases with zeros
        Checkout np.zeros function.
        Read the writeup to identify the right shapes for all.
        """
        self.W = np.zeros((out_features,in_features))
        self.b = np.zeros((out_features ,1))
        self.A = None
        self.N = None
        self.debug = debug

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output Z of linear layer with shape (N, C1)
        Read the writeup for implementation details
        """
        self.A = A  # TODO
        self.N = A.shape[0]  # TODO store the batch size of input
        self.Ones = np.ones((self.N,1))
        # Think how will self.Ones helps in the calculations and uncomment below
        Z = np.matmul(A, self.W.transpose()) + np.matmul(self.Ones, self.b.transpose())

        return Z

    def backward(self, dLdZ):

        dLdA = np.matmul(dLdZ , self.W)
        self.dLdW = np.matmul(dLdZ.transpose(), self.A) 
        self.dLdb = np.matmul(dLdZ.transpose(), self.Ones)

        if self.debug:
            
            self.dLdA = dLdA

        return dLdA
