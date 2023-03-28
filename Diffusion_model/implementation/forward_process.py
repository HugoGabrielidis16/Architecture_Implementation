import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image



class ForwardNoise:
    def __init__(self, 
                 n_step : int = 10000):
        
        self.n_step = n_step
        self.beta = torch.linspace(0.001, 0.02, n_step)

    def forward(self, x):
        eps = torch.randn_like(x) 
        x_t = [x]
        for t in range(self.n_step):
            """
            Sampling from gaussian distribution : mean + std*eps
            """
            
            mean = (1 - self.beta[t])**(0.5) * x_t[t]
            var = self.beta[t]
            x_t.append(mean + eps)
            #x_t.append(mean + var**(0.5)*eps)
        return(x_t)
    
    def visualize(self, x):
        x_t = self.forward(x)
        fig, ax = plt.subplots(1, self.n_step, figsize=(30, 30))
        for t in range(self.n_step):
            ax[t].imshow(x_t[t].permute(1, 2, 0))
        plt.show()


if __name__ == '__main__':
    forward_process = ForwardNoise(8)
    img = Image.open("images/dog.jpeg")
    transform = T.ToTensor()
    tensor_img = transform(img)
    forward_process.visualize(tensor_img)


