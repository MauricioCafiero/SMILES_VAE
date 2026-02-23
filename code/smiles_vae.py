import tensorflow as tf
import numpy as np
import pandas as pd
import deepchem as dc
import time
import transformers
from rdkit import Chem
import matplotlib.pyplot as plt
from rdkit.Chem import AllChem, Draw
from sklearn.model_selection import train_test_split
from deepchem.feat.smiles_tokenizer import SmilesTokenizer

def make_datasets(filename: str, smiles_column = 'SMILES'):
  '''
    Tokenizes a dataset and returns the input and target arrays.

      Args:
        filename: name of new dataset
        smiles_column: name of the smiles column
      Returns:
        fx: input array
        fy: target array
        VOCAB_SIZE: vocabulary size
        tokenizer: tokenizer object
        max_length: longest SMILES chain
  '''
  df = pd.read_csv(filename)

  Xa = []
  for smiles in df[smiles_column]:
    smiles = smiles.replace("[Na+].","").replace("[Cl-].","").replace(".[Cl-]","").replace(".[Na+]","")
    smiles = smiles.replace("[K+].","").replace("[Br-].","").replace(".[K+]","").replace(".[Br-]","")
    smiles = smiles.replace("[I-].","").replace(".[I-]","").replace("[Ca2+].","").replace(".[Ca2+]","")
    Xa.append(smiles)

  #===========================================================================================
  #featurize

  tokenizer=dc.feat.SmilesTokenizer(vocab_file="SMILES_VAE/data/vocab_305K.txt")
  featname="SMILES Tokenizer"

  fl = list(map(lambda x: tokenizer.encode(x),Xa))

  biggest = 1
  smallest = 200
  for i in range(len(fl)):
      temp = len(fl[i])
      if temp > biggest:
          biggest = temp
      if temp < smallest:
          smallest = temp

  print(biggest, smallest)

  string_length = smallest - 1
  max_length = biggest

  fl2 = list(map(lambda x: tokenizer.add_padding_tokens(x,max_length),fl))

  # fl2set=set()
  # for sublist in fl2:
  #   fl2set.update(sublist)
  # temp_vocab_size = len(fl2set)

  f = open("SMILES_VAE/data/vocab_305K.txt", "r")
  lines = f.readlines()
  f.close()
  VOCAB_SIZE = len(lines)
  print("Vocabulary size for this dataset: ",VOCAB_SIZE)

  x = []
  y = []
  i=0
  for string in fl2:
      x.append(string[0:max_length-1]) #string_length
      y.append(string[1:max_length]) #string_length+1

  x = np.array(x)
  y = np.array(y)
  print("Number of features and datapoints, targets: ",x.shape,y.shape)

  #===========================================================================================
  print("featurization done with: ",featname)

  fx = x
  fy = y

  return fx, fy, VOCAB_SIZE, tokenizer, max_length #fl2set

def trim_vocab(filename: str, tokens_to_remove: list, smiles_column = "SMILES"):
  '''
    trims entries from a SMILES list that contain tokens not found in the 
    Foundation model's vocabulary list. Also trims entries that are longer than the 
    Foundation model's context window. 
    
        Args:
            filename: a CSV file with the dataset to be trimmed
            tokens_to_remove: a set pf tokens to remove,  obtained from the test_vocab tool.
        Returns:
            None: a new CSV file is saved with the trimmed list.
  '''
  
  
  df = pd.read_csv(filename)

  Xa = []
  for smiles in df[smiles_column]:
    smiles = smiles.replace("[Na+].","").replace("[Cl-].","").replace(".[Cl-]","").replace(".[Na+]","")
    smiles = smiles.replace("[K+].","").replace("[Br-].","").replace(".[K+]","").replace(".[Br-]","")
    smiles = smiles.replace("[I-].","").replace(".[I-]","").replace("[Ca2+].","").replace(".[Ca2+]","")
    Xa.append(smiles)
  
  smiles_removed_tokens = []
  for i,smiles in enumerate(Xa):
    bad_list = [True if (token in smiles) else False for token in tokens_to_remove]
    if not any(bad_list):
      smiles_removed_tokens.append(smiles)   

  smiles_no_long = []
  for i,smiles in enumerate(smiles_removed_tokens):
    if len(smiles) <= 166:
      smiles_no_long.append(smiles)
  
  print(f"Removed {len(Xa) - len(smiles_no_long)} entries from the list!")
  
  new_dict = {"SMILES": smiles_no_long}
  new_df = pd.DataFrame(new_dict)
  new_df.to_csv(f'{filename.replace(".csv","")+"_trimmed.csv"}', index=False)
  print("New CSV file written!")
  
def test_vocab(filename: str, smiles_column = 'SMILES'):
  '''
    Tests the vocabulary of a new dataset against the foundation model vocabulary.
    Rejects if the new dataset has tokens not in the foundation model vocabulary, or if
    the context window is too large.

      Args:
        filename: name of new dataset
        smiles_column: name of the smiles column
      Returns:
        novel_items: list of tokens not in the foundation model vocabulary
  '''
  df = pd.read_csv(filename)

  Xa = []
  for smiles in df[smiles_column]:
    smiles = smiles.replace("[Na+].","").replace("[Cl-].","").replace(".[Cl-]","").replace(".[Na+]","")
    smiles = smiles.replace("[K+].","").replace("[Br-].","").replace(".[K+]","").replace(".[Br-]","")
    smiles = smiles.replace("[I-].","").replace(".[I-]","").replace("[Ca2+].","").replace(".[Ca2+]","")
    Xa.append(smiles)

  #===========================================================================================
  #featurize

  tokenizer=dc.feat.SmilesTokenizer(vocab_file="SMILES_VAE/data/vocab.txt")
  featname="SMILES Tokenizer"

  fl = list(map(lambda x: tokenizer.encode(x),Xa))

  biggest = 1
  smallest = 200
  for i in range(len(fl)):
      temp = len(fl[i])
      if temp > biggest:
          biggest = temp
      if temp < smallest:
          smallest = temp

  print(biggest, smallest)

  string_length = smallest - 1
  max_length = biggest

  fl2 = list(map(lambda x: tokenizer.add_padding_tokens(x,max_length),fl))

  fl2set=set()
  for sublist in fl2:
    fl2set.update(sublist)
  new_vocab_size = len(fl2set)
  print("New vocabulary size: ",new_vocab_size)

  f = open("SMILES_VAE/data/vocab_305K.txt", "r")
  raw_lines = f.readlines()
  f.close()
  VOCAB_SIZE = len(raw_lines)
  print("Vocabulary size for standard dataset: ",VOCAB_SIZE)

  lines = []
  for line in raw_lines:
    lines.append(line.replace("\n",""))

  novel_items = []
  for item in fl2set:
    item = tokenizer.decode([item])
    item = tokenizer.convert_tokens_to_string(item)
    item = item.replace(" ","")

    if item not in lines:
      print(f"{item} not in standard vocabulary")
      novel_items.append(item)

  if(len(novel_items) > 0):
    print("This dataset is not compatible with the Foundation model vocabulary")
  else:
    print("This dataset is compatible with the Foundation model vocabulary")

  if max_length > 166:
    print("This dataset's context window is not compatible with the Foundation model.")
  else:
    print("This dataset's context window is compatible with the Foundation model")

  return novel_items

class Sampling(tf.keras.layers.Layer):
    def call(self,inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch,dim))
        return z_mean + tf.exp(0.5*z_log_var)*epsilon

class KL_Loss_Layer(tf.keras.layers.Layer):
  '''
  '''
  def __init__(self, scale_ll: float = 0.0, **kwargs):
    super(KL_Loss_Layer, self).__init__(**kwargs)
    self.scale_ll = scale_ll

  def call(self, inputs):
    z_mean, z_log_var = inputs
    kl_divergence_per_sample = -0.5 * tf.reduce_sum(1 + z_log_var - tf.exp(z_log_var) - tf.square(z_mean), axis=-1)
    kl_loss = self.scale_ll * tf.reduce_mean(kl_divergence_per_sample)
    self.add_loss(kl_loss) # Add as an auxiliary loss to the model
    return z_mean

class VAE():
  '''
  A variational autoencoder model for SMILES strings.
  '''

  def __init__(self, emb_size: int = 256, latent_size: int = 256, num_layers: int = 2, num_units: int = 128, 
                      scale_ll: float = 0.000001, max_length: int = 105, vocab_size: int = 74):
    '''
    constructor for the VAE class.
    Args:
      emb_size: embedding size
      latent_size: latent size
      num_layers: number of GRU layers
      num_units: number of units in each GRU layer
      scale_ll: scale factor for the KL loss
      max_length: maximum length of a SMILES string
      vocab_size: size of the vocabulary
    '''
    self.emb_size = emb_size
    self.latent_size = latent_size
    self.num_layers = num_layers
    self.num_units = num_units
    self.scale_ll = scale_ll
    self.max_length = max_length
    self.vocab_size = vocab_size

  def make_vae(self):
    '''
    Creates the encoder, decoder, and autoencoder.
    Also creates the sampling layer and the KL loss layer.
    '''
    #define encoder
    encoder_input = tf.keras.layers.Input(shape=[self.max_length], name = "encoder_input")
    x = tf.keras.layers.Embedding(input_dim=self.vocab_size,output_dim=self.emb_size)(encoder_input)
    for _ in range(self.num_layers):
      x = tf.keras.layers.GRU(self.num_units,return_sequences=True)(x)

    #get shape and flatten
    shape_before_flattening = x.shape[1:]
    x = tf.keras.layers.Flatten()(x)

    # create quantities needed for sampling layer
    z_mean = tf.keras.layers.Dense(self.latent_size,name = "z_mean")(x)
    z_log_var = tf.keras.layers.Dense(self.latent_size,name = "z_log_var")(x)
    z = Sampling()([z_mean,z_log_var])

    self.encoder = tf.keras.models.Model(encoder_input,[z_mean,z_log_var,z], name = "encoder")

    decoder_input = tf.keras.layers.Input(shape=(self.latent_size,), name = "decoder_input")
    x = tf.keras.layers.Dense(np.prod(shape_before_flattening))(decoder_input)
    x = tf.keras.layers.Reshape(shape_before_flattening)(x)
    for _ in range(self.num_layers):
      x = tf.keras.layers.GRU(self.num_units,return_sequences=True)(x)
    decoder_output = tf.keras.layers.Dense(self.vocab_size,activation="softmax")(x)

    self.decoder = tf.keras.models.Model(decoder_input,decoder_output)

    outputs = self.decoder(z)

    # Instantiate KL_Loss_Layer and add it to the model's graph. This layer will automatically add the KL loss.
    kl_loss_output = KL_Loss_Layer(scale_ll = self.scale_ll, name='kl_divergence_layer')([z_mean, z_log_var])

    self.autoencoder = tf.keras.models.Model(encoder_input, outputs)

  def compile_vae(self, X, y, epochs: int = 30, batch_size: int =128, optimizer: str = 'Adam'):
    '''
    Sets up the VAE model for training.
    Args:
      X: input data
      y: target data
      epochs: number of epochs to train for
      batch_size: batch size
      optimizer: optimizer to use
    '''
    self.epochs = epochs
    self.batch_size = batch_size
    self.X = X
    self.y = y

    if optimizer == 'Adam':
      optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
  
    self.autoencoder.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
  
  def train_vae(self):
    '''
    Trains the VAE model.
    Returns:
      history: the training history
    '''
    # checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(r"ConvAE/ConvAE.keras",monitor="val_accuracy",save_best_only=True)
    history = self.autoencoder.fit(self.X, self.X, epochs=self.epochs, verbose=2, validation_split=0.1) #,callbacks=[checkpoint_cb])
    
    return history
  
  def test_vae(self, smiles_list: list, tokenizer, test_size: int = 20):
    '''
    '''

    rand_set = np.random.randint(0,len(self.X),size=test_size) #int(0.1*len(dataset.X)))
    Xs_raw = []
    for i in rand_set:
        Xs_raw.append(smiles_list[i])
    print(f'the length of Xs is: {len(Xs_raw)}')

    Xs = list(map(lambda x: tokenizer.encode(x),Xs_raw))
    Xs = list(map(lambda x: tokenizer.add_padding_tokens(x,self.max_length),Xs))

    test_array = np.array(Xs)

    results = self.autoencoder.predict(test_array)
    proba = np.empty((len(results),self.max_length,self.vocab_size))

    new_mols_ids = []
    for mol in range(len(results)):
      pred = []
      for i in range(self.max_length):
          proba[mol,i,:] = results[mol,i,:]
          # append only the value from each tensor
          pred.append(tf.argmax(proba[mol,i,:]).numpy())
      new_mols_ids.append(pred)

    new_mols = [tokenizer.decode(mol) for mol in new_mols_ids]
    new_mols = [mol.replace(" ","").replace("[CLS]","").replace("[SEP]","").replace("[PAD]","") for mol in new_mols]

    mols = []
    legends = []
    hits = 0
    for old_molecule, new_molecule in zip(Xs_raw,new_mols):
        if old_molecule == new_molecule:
          hits += 1
        mols.append(Chem.MolFromSmiles(old_molecule))
        mols.append(Chem.MolFromSmiles(new_molecule))
        legends.append('Input Molecule')
        legends.append('Reconstructed Molecule')
        
    losses = len(Xs_raw) - hits
    print(f'Hits: {hits}')
    print(f'Losses: {losses}')
    print(f'Accuracy: {hits/len(Xs_raw)}')
    img = Draw.MolsToGridImage(mols=mols, legends=legends,molsPerRow=2,maxMols=500)

    return img
  
  def generate(self, tokenizer, num_samples: int = 50):
    '''
    Generates new SMILES strings by sampling from the latent space and decoding.
    Args:
        num_samples: number of new SMILES strings to generate
        tokenizer: tokenizer used for encoding/decoding SMILES strings
    Returns:
        img: a grid image of the generated molecules
    '''
    z_gen = np.random.normal(size=(num_samples, self.emb_size))

    recons_embed = self.decoder.predict(z_gen)
    print(recons_embed.shape)

    proba = np.empty((len(recons_embed),self.max_length,self.vocab_size))

    new_mols_ids = []
    for mol in range(len(recons_embed)):
      pred = []
      for i in range(self.max_length):
          proba[mol,i,:] = recons_embed[mol,i,:]
          # append only the value from each tensor
          pred.append(tf.argmax(proba[mol,i,:]).numpy())
      new_mols_ids.append(pred)

    new_mols = [tokenizer.decode(mol) for mol in new_mols_ids]
    self.new_mols = [mol.replace(" ","").replace("[CLS]","").replace("[SEP]","").replace("[PAD]","") for mol in new_mols]

    mols = []
    legends = []
    hits = 0
    misses = 0
    for new_molecule in self.new_mols:
      mol_temp = Chem.MolFromSmiles(new_molecule)
      if mol_temp != None:
        mols.append(mol_temp)
        legends.append(new_molecule)
        hits += 1
      else:
        misses += 1
    
    print(f'Hits: {hits}')
    print(f'Misses: {misses}')
    try:
      print(f'Accuracy: {hits/(hits+misses)}')
      img = Draw.MolsToGridImage(mols=mols, legends=legends,molsPerRow=2,maxMols=50)
    except:
      img = None

    return img