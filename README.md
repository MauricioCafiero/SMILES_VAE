# SMILES Variational Autoencoder
- GRU layers for the encoder and decoder; number of layers variable.
- Tokenize and encode SMILES from a CSV file
- Test full autoencoder and
- Generate new molecules with decoder.

## ARCHIVE: Chemistry Autoencoders and Generative Models
Autoencoders, encoders and decoders, and generative adversarial networks form the basis of many modern generative ML/AI models. This project is a set of basic 
chemistry autoencoders and generative models that can be used as a starting point for building other ML models. Image models use a SMILES to image featurizer which embeds 
molecular information into a 4-channel image.

## This project includes:
- A SMILES string autoencoder, using GRU layers.
- A 4-channel molecular graph image autoencoder using Dense layers.
- A 4-channel molecular graph image autoencoder using Convolutional layers.
- A 4-channel variational molecular graph image autoencoder using Convolutional layers.
- a 4 channel molecular graph image generative adversarial network using Convolutional layers.
- a 4 channel molecular graph image Wasserstein generative adversarial network with gradient penalty using Convolutional layers.
- a 4 channel molecular graph image Pixel CNN based on the Tensorflow distributions implementation.
