import numpy as np
import scipy

class Identity:

    def forward(self, Z):

        self.A = Z

        return self.A

    def backward(self, dLdA):

        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ

        return dLdZ


class Sigmoid:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Sigmoid.
    """
    def forward(self, Z):

        self.A = 1.0/(1.0 + np.exp(-Z))

        return self.A

    def backward(self, dLdA):

        dAdZ = self.A * (1.0 -  self.A)
        dLdZ = dLdA * dAdZ
        return dLdZ


class Tanh:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Tanh.
    """
    def forward(self, Z):

        self.A = (np.exp(Z) - np.exp(-Z))/(np.exp(Z) + np.exp(-Z))

        return self.A

    def backward(self, dLdA):

        dAdZ = 1 - self.A ** 2
        dLdZ = dLdA * dAdZ

        return dLdZ

class ReLU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on ReLU.
    """
    def forward(self, Z):

        self.A = np.maximum(0, Z)

        return self.A

    def backward(self, dLdA):

        dAdZ = (self.A > 0).astype(float)
        dLdZ = dLdA * dAdZ

        return dLdZ


class GELU:
    """
    GELU activation function.
    """
    def forward(self, Z):
        # Compute the error function part of GELU
        self.error = 0.5 * (1 + scipy.special.erf(Z / np.sqrt(2.0)))
        
        # Store the input and compute the GELU activation
        self.Z = Z
        self.A = Z * self.error
        
        return self.A

    def backward(self, dLdA):
        # Compute the gradient of the GELU activation
        dAdZ = self.error + (self.Z / np.sqrt(2.0 * np.pi)) * np.exp(-self.Z ** 2 / 2.0)
        
        # Compute the gradient of the loss with respect to Z
        dLdZ = dLdA * dAdZ
        
        return dLdZ
    
class GELU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on GELU.
    """
    def forward(self, Z):
        self.error = 0.5 * (1 +  scipy.special.erf(Z / np.sqrt(2.0)))
        self.Z = Z
        self.A = Z * self.error
        
        return self.A

    def backward(self, dLdA):

        dAdZ = self.error + (self.Z / np.sqrt( 2.0 * np.pi) )* np.exp(-self.Z ** 2 / 2.0)
        # Compute the gradient of the loss with respect to Z
        dLdZ = dLdA * dAdZ
        return dLdZ

class Softmax:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Softmax.
    """

    def forward(self, Z):
        """
        Remember that Softmax does not act element-wise.
        It will use an entire row of Z to compute an output element.
        """
        # Subtract max(Z) to improve numerical stability
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        
        # Compute the Softmax activation
        self.A = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        
        return self.A
    
    def backward(self, dLdA):

        # Calculate the batch size and number of features
        N, C = self.A.shape


        # Initialize the final output dLdZ with all zeros. Refer to the writeup and think about the shape.
        dLdZ = dLdZ = np.zeros((N, C))

        # Fill dLdZ one data point (row) at a time
        for i in range(N):

            # Initialize the Jacobian with all zeros.
            J = J = np.zeros((C, C))

            # Fill the Jacobian matrix according to the conditions described in the writeup
            for m in range(C):
                for n in range(C):
                    if n == m:
                        J[m,n] = self.A[i,m] * (1 - self.A[i,n])
                    else:
                        J[m, n] = -self.A[i, m] * self.A[i, n]

            # Calculate the derivative of the loss with respect to the i-th input
            dLdZ[i,:] = np.dot(dLdA[i, :], J)

        return dLdZ
