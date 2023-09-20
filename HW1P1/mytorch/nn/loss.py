import numpy as np


class MSELoss:

    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)

        """

        self.A = A
        self.Y = Y
        self.N, self.C = A.shape
        se = (self.A - self.Y) * (self.A - self.Y)
        sse = np.sum(se)
        mse = sse/ (self.N * self.C)

        return mse

    def backward(self):

        dLdA = 2 * (self.A - self.Y) / (self.N * self.C)

        return dLdA


class CrossEntropyLoss:

    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        Refer the the writeup to determine the shapes of all the variables.
        Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        N, C = self.A.shape

        Ones_C = np.ones((C, 1))
        Ones_N = np.ones((N, 1))

        exp_A = np.exp(self.A - np.max(self.A, axis=1, keepdims=True))
        
        # Compute the Softmax activation
        self.softmax = exp_A / np.sum(exp_A, axis=1, keepdims=True)
       
        crossentropy = np.matmul(-Y  * np.log(self.softmax), Ones_C)
        sum_crossentropy = np.matmul(Ones_N.transpose(), crossentropy)
        L = sum_crossentropy / N

        return L

    def backward(self):

        dLdA = self.softmax - self.Y

        return dLdA / self.A.shape[0]
