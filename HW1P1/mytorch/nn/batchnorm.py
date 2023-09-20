import numpy as np


class BatchNorm1d:

    def __init__(self, num_features, alpha=0.9):

        self.alpha = alpha
        self.eps = 1e-8

        self.BW = np.ones((1, num_features))
        self.Bb = np.zeros((1, num_features))
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))

        # Running mean and variance, updated during training, used during
        # inference
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the
        training phase of the problem or the inference phase.
        So see what values you need to recompute when eval is False.
        """
        self.Z = Z
        self.N = self.Z.shape[0]
        self.M = np.mean(self.Z, axis=0, keepdims=True)
        self.V = np.var(self.Z, axis=0, keepdims=True)

        if eval == False:
            # training mode
            self.NZ = (self.Z - self.M)/ np.sqrt(self.V + self.eps)
            self.BZ = self.BW * self.NZ + self.Bb

            self.running_M = self.alpha * self.running_M + (1 - self.alpha) * self.M
            self.running_V = self.alpha * self.running_V + (1 - self.alpha) * self.V
        else:
            # inference mode
            
            self.NZ = (self.Z - self.running_M) / np.sqrt(self.running_V + self.eps)
            self.BZ = self.BW * self.NZ + self.Bb


        return self.BZ

    # def backward(self, dLdBZ):

    #     self.dLdBW = None  # TODO
    #     self.dLdBb = None  # TODO

    #     dLdNZ = None  # TODO
    #     dLdV = None  # TODO
    #     dLdM = None  # TODO

    #     dLdZ = None  # TODO

    #     return NotImplemented

    def backward(self, dLdBZ):
        """
        Compute the backward pass.
        """
        # Gradients of loss with respect to scale and shift parameters
        self.dLdBW = np.sum(dLdBZ * self.NZ, axis=0, keepdims=True)
        self.dLdBb = np.sum(dLdBZ, axis=0, keepdims=True)

        # Gradient of loss with respect to normalized Z
        dLdNZ = dLdBZ * self.BW

        # Gradient of loss with respect to variance
        dLdV = np.sum(dLdNZ * (self.Z - self.M) * -0.5 * np.power(self.V + self.eps, -1.5), axis=0, keepdims=True)

        # Gradient of loss with respect to mean
        dLdM = np.sum(dLdNZ * -1 / np.sqrt(self.V + self.eps), axis=0, keepdims=True) + dLdV * np.sum(-2 * (self.Z - self.M), axis=0, keepdims=True) / self.N

        # Gradient of loss with respect to Z
        dLdZ = dLdNZ / np.sqrt(self.V + self.eps) + dLdV * 2 * (self.Z - self.M) / self.N + dLdM / self.N

        return dLdZ
