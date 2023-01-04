"""
The loss function of the VAE is the sum two loss function : 
L = L1 + L2 

We have L1 : the reconstruction loss (MSE or binary cross entropy), distance between our original image and the 
reconstructed one
L2 : KL (Kullback Leibler divergence) : Is there to ensure our latent distribution will converge to a gaussian 
distribution centered on 0 with a sd of 1. We use this because of the nice property of the gaussian which set up a good expression
for the loss

"""
import math
import torch


def KL_loss(mean, var):
    """
    Return the KL loss between our predicted normal distribution and a standard normal distribution.

    Arg
    ------
    mean (float): mean of our distribution.
    var (float): variance of our distribution.

    Returns
    ------
    loss (float) : result of the KL-loss.
    """
    loss = -(1 / 2) * torch.sum(1 + var - mean**2 - torch.exp(var))
    loss = loss.mean()
    return loss


def MSE_loss(y_pred, y_true, batch_size=32):
    """
    Take as input two batched tensor of images and return the mean of the MSE loss over the batch dimension

    Args
    ----
    y_pred (Tensor) : Image predicted by the model.
    y_true (Tensor) : Ground truth image.
    batch_size (int) : Batch_size.
    """

    y_pred_flat = y_pred.view(batch_size, -1)
    y_true_flat = y_true.view(batch_size, -1)

    assert y_pred_flat.shape[1] == y_true_flat.shape[1]
    size = y_pred_flat.shape[1]
    loss = (y_pred_flat - y_true_flat) ** 2
    loss_for_each_batch = (
        loss.sum(axis=1) / size
    )  # Sum over each pixels and divide by the size
    mean_loss = loss_for_each_batch.mean()  # Average over batch dimension
    return mean_loss


if __name__ == "__main__":
    mean = torch.randn(6, 100)
    var = torch.randn(6, 100)
    print(KL_loss(mean, var))
