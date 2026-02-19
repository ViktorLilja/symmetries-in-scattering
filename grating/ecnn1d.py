import torch
import math
import numpy as np

def getOrderedIrreps(flip_repr):
    irreps, Q_inv = np.linalg.eig(flip_repr)

    idx = irreps.argsort()[::-1]
    irreps = irreps[idx]
    Q_inv = Q_inv[:,idx]

    return irreps, Q_inv

class FlipEquivariantKernel(torch.nn.Module):
    """
    Generates flip equivariant kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, in_flip_repr, out_flip_repr):
        super().__init__()

        self.in_channels   = in_channels
        self.out_channels  = out_channels
        self.kernel_size   = kernel_size
        self.in_flip_repr  = torch.tensor(in_flip_repr, dtype=torch.float)
        self.out_flip_repr = torch.tensor(out_flip_repr, dtype=torch.float)

        self.in_dim  = self.in_flip_repr.shape[0]
        self.out_dim = self.out_flip_repr.shape[0]

        # Check representations are valid for the reflection group
        assert torch.allclose(self.in_flip_repr @ self.in_flip_repr, torch.eye(self.in_dim)); "Invalid input representation"
        assert torch.allclose(self.out_flip_repr @ self.out_flip_repr, torch.eye(self.out_dim)); "Invalid output representation"

        # Construct tensor product of representations and calculate
        # tensor change of basis matrix and irreps vector
        self.tensor_flip_repr = np.kron(self.in_flip_repr.numpy(), self.out_flip_repr.numpy())
        irreps, Q_inv = getOrderedIrreps(self.tensor_flip_repr)
        self.irreps = torch.tensor(irreps)

        # To make sure Q_inv is on the same device as the kernel
        # use this instead of self.Q_inv = torch.tensor(Q_inv, dtype=torch.float)
        self.register_buffer("Q_inv", torch.tensor(Q_inv, dtype=torch.float), persistent=False) 

        # Make sure all irreps are 1 or -1, as they must for the reflection group.
        assert torch.allclose(torch.abs(self.irreps), torch.tensor(1.), atol=1e-6)

        # Calculate the number of odd and even irreps
        self.num_even_irreps = torch.sum(torch.abs(self.irreps - torch.tensor(1.))<1e-6, dtype=torch.int)
        self.num_odd_irreps  = torch.sum(torch.abs(self.irreps + torch.tensor(1.))<1e-6, dtype=torch.int)
        assert self.num_even_irreps + self.num_odd_irreps == self.in_dim * self.out_dim

        # Calculate the number of free parameters in each kernel element.
        # For example, an odd kernel of size 5 must be on the form
        # [-b, -a, 0, a, b], so it has 2 free parameters
        self.even_params = int((self.kernel_size+1)//2) 
        self.odd_params  = int(self.kernel_size//2)

        # Construct the tensors containing the trainable parameters and matrices
        # that are used to calculate the full kernels from the half kernels by:
        # [-b] = [ 0 -1] [a]
        # [-a]   [-1  0] [b]
        # [ 0]   [ 0  0]
        # [ a]   [ 1  0]
        # [ b]   [ 0  1]
        if self.num_even_irreps > 0:
            self.weight_even = torch.nn.Parameter(torch.empty((
                self.out_channels,
                self.in_channels,
                self.num_even_irreps,
                self.even_params,
            ), dtype=torch.float))

            self.register_buffer(
                "_even_expansion_matrix", 
                torch.zeros((self.kernel_size, self.even_params), dtype=torch.float)
            )
            self._even_expansion_matrix[-self.even_params:,:] = torch.eye(self.even_params)
            self._even_expansion_matrix[:self.even_params,:]  = torch.flip(torch.eye(self.even_params), dims=(1,))

        if self.num_odd_irreps > 0 and self.kernel_size > 1:
            self.weight_odd = torch.nn.Parameter(torch.empty((
                self.out_channels,
                self.in_channels,
                self.num_odd_irreps,
                self.odd_params,
            ), dtype=torch.float))

            self.register_buffer(
                "_odd_expansion_matrix", 
                torch.zeros((self.kernel_size, self.odd_params), dtype=torch.float)
            )
            self._odd_expansion_matrix[-self.odd_params:,:] =  torch.eye(self.odd_params)
            self._odd_expansion_matrix[:self.odd_params,:]  = -torch.flip(torch.eye(self.odd_params), dims=(1,))

    def sample(self):
        """
        Construct the full kernel of shape 
        [out_channels, out_dim, in_channels, in_dim, kernel_size] 
        satisfying the equivariance constraint.

        This function is called sample to match the nomenclature in the escnn
        library, where kernels are expanded in a continous basis and sampled
        on a grid. Here, however, the kernel elements on the grid are stored
        directly.
        """

        kernel = torch.zeros((
            self.out_channels,
            self.in_channels,
            self.in_dim * self.out_dim,
            self.kernel_size,
        ), dtype=torch.float, device=self.Q_inv.device)

        # Populate the kernel with the free parameters
        if self.num_even_irreps > 0:
            kernel[:,:,:self.num_even_irreps,:] = torch.einsum('di, abci -> abcd', self._even_expansion_matrix, self.weight_even)
        if self.num_odd_irreps > 0 and self.kernel_size > 1:
            kernel[:,:,self.num_even_irreps:,:] =  torch.einsum('di, abci -> abcd', self._odd_expansion_matrix,  self.weight_odd)

        # Change to the non-irreps basis
        kernel = torch.einsum('ci, abid -> abcd', self.Q_inv, kernel)

        # Reshape from [out_channels, in_channels, in_dim * out_dim, kernel_size]
        # to           [out_channels, in_channels, in_dim, out_dim, kernel_size]
        # to           [out_channels, out_dim, in_channels, in_dim, kernel_size]

        return kernel.view(
            self.out_channels, self.in_channels, self.in_dim, self.out_dim, self.kernel_size
        ).permute(0, 3, 1, 2, 4)
    

class GConv1d(torch.nn.Module):
    """
    Group equivariant convolution for the reflection group on a 1D base space.

    Input: feature field of shape
    [batch_size,channels,in_dim,spatial_dimension]
    transforming according to representation given by in_flip_repr of shape
    (in_dim, in_dim)

    Output: feature field of shape
    [batch_size,channels,out_dim,spatial_dimension]
    transforming according to representation given by out_flip_repr of shape
    (out_dim, out_dim)

    """

    def __init__(
            self, 
            in_channels, 
            out_channels, 
            kernel_size,
            in_flip_repr,
            out_flip_repr,
            padding=None,
            padding_mode='constant',
            stride=1, 
            dilation=1, 
            bias=True, 
        ):
        super().__init__()

        self.padding      = padding
        self.padding_mode = padding_mode
        self.stride       = stride
        self.dilation     = dilation
        self.bias         = bias

        # If only one number is given, assume padding is symmetric
        if padding != None:
            if not hasattr(self.padding, '__iter__'):
                self.padding = [self.padding, self.padding]
            assert len(self.padding) == 2

        self.kernel = FlipEquivariantKernel(
            in_channels, out_channels, kernel_size, in_flip_repr, out_flip_repr
        )

        self.cached_kernel = self.kernel.sample()
        self.cached_kernel = self.cached_kernel.reshape(
            self.cached_kernel.shape[0] * self.cached_kernel.shape[1],
            self.cached_kernel.shape[2] * self.cached_kernel.shape[3],
            self.cached_kernel.shape[4],
        )

        if bias:
            assert self.supports_pointwise_nonlinearity(); "Bias is not supported for this output representation."
            # To ensure equivariance, the bias is shared across both the group
            # dimension (2) and the spatial dimension (3).
            self.bias = torch.nn.Parameter(torch.zeros((
                1, self.kernel.out_channels, 1, 1
            )))
        else:
            self.bias = None
        
        self.reset_parameters()

    def supports_pointwise_nonlinearity(self) -> bool:
        # Pointwise nonlinearities are supported only if the output features
        # transforms by a permutation, i.e. the output representation
        # has only zeros and ones.
        tol = 1e-6
        return torch.all(
            (torch.abs(self.kernel.out_flip_repr-1) < tol) | 
            (torch.abs(self.kernel.out_flip_repr) < tol)
        )

    def reset_parameters(self) -> None:
        for tensor in self.kernel.parameters():
            torch.nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))
        if self.bias is not None:
            for tensor in self.kernel.parameters():
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(tensor)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(self.bias, -bound, bound)


    def forward(self, x):

        # Construct the full kernel from the weights
        if self.training:
            self.cached_kernel = self.kernel.sample()
            self.cached_kernel = self.cached_kernel.reshape(
                self.cached_kernel.shape[0] * self.cached_kernel.shape[1],
                self.cached_kernel.shape[2] * self.cached_kernel.shape[3],
                self.cached_kernel.shape[4],
            )

        # Group convolution is implemented using conv1d by temporarily
        # combining the channel dimension and the group dimension of
        # both the input feature fields and the kernel.
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        if self.padding != None:
            x = torch.nn.functional.pad(x, self.padding, mode=self.padding_mode)

        x = torch.nn.functional.conv1d(
            input=x,
            weight=self.cached_kernel,
            stride=self.stride,
            dilation=self.dilation
        )
        x = x.view(x.shape[0], self.kernel.out_channels, self.kernel.out_dim, x.shape[2])

        # Optional channel-wise bias
        if self.bias != None: x = x + self.bias

        return x