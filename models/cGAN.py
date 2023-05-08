import torch
import torch.nn as nn


__all__ = ['Generator', 'Discriminator']

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Used to create a label embedding.
        self.label_embedding = nn.Embedding(10, 10)
        
        # Used to create the 1024 channel deep tensor, as in the DCGAN paper.
        self.l1 = nn.ConvTranspose2d(110, 1024, 4, 1, 0, bias=False) # (3, 110, 1, 1) -> (3, 1024, 4, 4)
        
        # (3, 110, 1, 1) -> (3, 1, 32, 32)
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )
    def forward(self, x, label):
        label = self.label_embedding(label.long())
        label = label.unsqueeze(2) 
        label = torch.swapaxes(label, 1, 3)

        x = torch.cat([x, label], 1)

        # Pass through the initial layer which doesn't have bias or activation.
        x = self.l1(x)

        # Pass through the convolutional layers.
        x = self.conv_blocks(x)
        
        # Return in shape (batch_size, 784)
        return torch.flatten(x[:,:,2:-2,2:-2].squeeze(1), start_dim = 1)

g = Generator()
x = g(torch.randn((3, 100,1,1)), torch.Tensor([[1], [2], [3]]).long())
x.shape


class Discriminator(nn.Module):
    def __init__(self, d_input_dim=28*28, n_class=10):
        super(Discriminator, self).__init__()
        
        # Used to create a label embedding.
        self.label_embedding = nn.Embedding(n_class, 50)
        self.embedding_layer = nn.Linear(50, 784)

        
        # A tensor of shape (batch_size, 2, 28, 28) is input into
        # six convolutional layers outputting a tensor of 
        # shape (batch_size, 1, 1, 1)
        self.model = nn.Sequential(
            nn.Conv2d(2,  32, 3, padding=3),
            # nn.Dropout2d(0.2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32,  64, 4, 2, 1, bias=True),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64,  128, 4, 2, 1, bias=True),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128,  256, 4, 2, 1, bias=True),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256,  512, 4, 2, 1, bias=True),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512,  1, 4, 2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    # forward method
    def forward(self, x, label):
        x = x.view(-1, 1, 28, 28)

        # Generate embedding.
        label = self.label_embedding(label)
        label_output = self.embedding_layer(label)
        label_output = label_output.view(-1, 1, 28, 28)
        
        # Concatenate the image and label embedding.
        x = torch.cat([x, label_output], 1)

        # Pass x into the model.
        x = self.model(x)

        # remove unnecessary dimensions.
        x = torch.flatten(x, start_dim = 1)

        # Return predicted value between 0 and 1.
        return torch.sigmoid(x)


