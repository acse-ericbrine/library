import torch
import torch.nn as nn


__all__ = ['Conditional_VAE_Encoder', 'Conditional_VAE_Decoder', 'Conditional_VAE']

class Conditional_VAE_Encoder(nn.Module):
    def __init__(self, latent_input_dim=256):
        super(Conditional_VAE_Encoder, self).__init__()

        # Used to create an embedding of the label.
        self.label_embedding = nn.Embedding(10, 50)
        self.embedding_layer = nn.Linear(50, 784)        
        
        # Two convolutional layers with a ReLU activation.
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(2, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.ReLU(),
        )

        self.out_layer = nn.Linear(3200, latent_input_dim)

    def forward(self, x, label):
        x = x.view(-1, 1, 28, 28)

        # create a (batch_size, 1, 28, 28) embedding of the label
        # that can be concatenated with x as a second channel.
        label = self.label_embedding(label)
        label_output = self.embedding_layer(label)
        label_output = label_output.view(-1, 1, 28, 28)

        x = torch.cat([x, label_output], 1)

        x = self.conv_blocks(x)

        # x is (batch_size, 128, 5, 5)
        x = torch.flatten(x, start_dim=1)

        # Shape goes from (batch_size, 3200) to (batch_size, latent_input_dim)
        x = self.out_layer(x)
        return x

class Conditional_VAE_Decoder(nn.Module):
  def __init__(self, latent_output_dim=256, n_class=10, embedding_dim=10, img_dim=(28,28)):
    super(Conditional_VAE_Decoder, self).__init__()
    '''
    Conditional Decode class (latent, label -> reconstructed image).
    '''
    self.img_dim = img_dim

    # The input to the first layer will be the concatenated
    # latent vector and label embedding.
    self.layer1 = nn.Linear(latent_output_dim + embedding_dim, 256)
    self.layer2 = nn.Linear(256, 512)
    self.layer3 = nn.Linear(512, 1024)

    # Will have 784 which can be reshaped into the image dimensions.
    self.layer4 = nn.Linear(1024, img_dim[0] * img_dim[1]) 

    # ReLU function is used for activation
    self.activation = nn.ReLU()
    
    # Sigmoid is used as activation for the output
    self.activationOut = nn.Sigmoid()

    # Used to create a label embedding
    self.label_embedding = nn.Embedding(n_class, embedding_dim)

    self.dropout = nn.Dropout(p=0.2)

  def forward(self, z, label):
    '''
    z: [float] a sample from the latent variable
    '''

    # Create label embedding
    label = self.label_embedding(label)

    # add dimension so it can be concatenated with z
    label = label.squeeze(1) 

    # concantenate label and z
    z = torch.cat([z, label], 1)

    # pass through first layer and activation
    z = self.activation(self.layer1(z))
    z = self.activation(self.dropout(self.layer2(z)))
    z = self.activation(self.dropout(self.layer3(z)))
    z = self.activationOut(self.layer4(z))
    return  z.reshape((-1,1,self.img_dim[0],self.img_dim[1]))

class Conditional_VAE(nn.Module):
  def __init__(self, latent_dim, latent_output_dim=256, device='cpu'):
    '''
    Conditional VAE class (img, label -> reconstructed image).
    '''
    super(Conditional_VAE, self).__init__()

    self.device = device

    # Instantiate encoder and decoder
    self.encoder = Conditional_VAE_Encoder(latent_input_dim=latent_output_dim)
    self.decoder = Conditional_VAE_Decoder(latent_output_dim=latent_output_dim)

    # Create a linear layer for Mu
    # self.layerMu = nn.Linear(2048, latent_dim)
    self.layerMu = nn.Linear(latent_output_dim, latent_dim)

    # Create a linear layer for Sigma
    # self.layerSig = nn.Linear(2048, latent_dim)
    self.layerSig = nn.Linear(latent_output_dim, latent_dim)

    # Create a normal distribution to sample from
    self.distribution = torch.distributions.Normal(0, 1)

    # Output layer and activation for latent space
    self.latentOut = nn.Linear(latent_dim, latent_output_dim)
    self.activationOut = nn.ReLU()

  def vae_latent_space(self, x):

    # Input the encoder output into Mu and Sigma layers
    # to obtain Mu and Sigma.
    mu = self.layerMu(x)
    sigma = torch.exp(self.layerSig(x))

    z = mu + sigma * self.distribution.sample(mu.shape).to(self.device)
    kl_div = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
    return z, kl_div

  def forward(self, x, label):

    # Encode x and the label
    x = self.encoder(x, label)

    # Sample the latent space
    z, kl_div = self.vae_latent_space(x)
    z = self.activationOut(self.latentOut(z))

    # Decode z and the label and return the result along with
    # the KL divergence used in training
    return self.decoder(z, label), kl_div
