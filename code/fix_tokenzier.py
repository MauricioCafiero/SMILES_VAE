from smiles_tokenizer import SMILESTokenizer
import pandas as pd

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

  tokenizer=SMILESTokenizer(vocab_file="../data/vocab.txt")
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

  #print(fl2set)
  new_vocab_size = len(fl2set)
  print("New vocabulary size: ",new_vocab_size)

  f = open("../data/vocab_305K.txt", "r")
  raw_lines = f.readlines()
  f.close()
  VOCAB_SIZE = len(raw_lines)
  print("Vocabulary size for standard dataset: ",VOCAB_SIZE)

  lines = []
  for line in raw_lines:
    lines.append(line.replace("\n",""))

  novel_items = []
  for item in fl2set:
    # Convert token ID to token string directly
    token = tokenizer.ids_to_tokens.get(int(item), tokenizer.unk_token)
    print(f"Checking token: {token}")

    if token not in lines:
      print(f"{token} not in standard vocabulary")
      novel_items.append(token)

  if(len(novel_items) > 0):
    print("This dataset is not compatible with the Foundation model vocabulary")
  else:
    print("This dataset is compatible with the Foundation model vocabulary")

  if max_length > 166:
    print("This dataset's context window is not compatible with the Foundation model.")
  else:
    print("This dataset's context window is compatible with the Foundation model")

  return novel_items

new_tokens = test_vocab('../data/ZN15_55K.csv', smiles_column = 'SMILES')

print("Novel tokens found: ", new_tokens)