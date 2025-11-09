import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader

"""
 Dataset is a list of tensors; each tensor holds tokens from tokenized text - tensor size is max_length
"""
class GPTDataset(Dataset):

   def __init__(self, txt, tokenizer, max_length, stride):
      """
       create python lists to hold tensors with input/target tokens
       each tensor size is max_length (# of tokens or context)
       stride is the # of tokens we shift to the right in tokenize text to compute next tensor
      """
      self.input_tensors = []
      self.target_tensors = [] 

      """
       compute tokenize version of the text
      """
      tokens = tokenizer.encode(txt)
      tokens = tokens[:50]
      # print("token len", len(tokens))

      """
       iterate the tokenized text to compute the input/target tensors of size max_length
       stride - controls the number of tokens we shift to the right on each iteration to create next tensor
      """
      for i in range(0, len(tokens) - max_length, stride):
         input_tokens = tokens[i:i+max_length]
         # to compute target we shift input 1 token to the right
         target_tokens = tokens[i+1: i+max_length + 1]
         self.input_tensors.append(torch.tensor(input_tokens))
         self.target_tensors.append(torch.tensor(target_tokens))
         # print("input:  ", i, i+max_length, input_tokens)
         # print("target: ", i+1, i+max_length+1,target_tokens)

   def __len__(self):
      """
       returns number of tensors in dataset
      """
      return len(self.input_tensors)

   def __getitem__(self, i):
      """
       returns a pair of input/target tensors in the set
      """
      return self.input_tensors[i], self.target_tensors[i] 

def create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
	tokenizer = tiktoken.get_encoding("gpt2")
	dataset = GPTDataset(txt, tokenizer, max_length, stride) # set of tensors with tokens
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
	return dataloader


"""
 the embedding layer is a matrix of random numbers each row representing a token (word) in our vocabulary and each column a dimension to describre the token (word)
"""
# test version: voca_size = 6 # number of tokens in our vocabulary (roughly # of words) 50k voca size for BPE tokenizer
voca_size = 50257
# test version: output_dim = 3 # number of dimensions to describe each token (word) chatGPT uses 12288 dimensions
output_dim = 256
torch.manual_seed(123) # hard coded side will return always same "random" #s
voca_embedding_layer = torch.nn.Embedding(voca_size, output_dim)
print("voca_embedding_layer:\n", voca_embedding_layer, "\n")


"""
 the token embeddings is a matrix of random numbers we compute the matrix by taking the token id and using it to grab an embedding vector from the embedding layer; we use that vector to represent the token (the embedding vector contains several dimensions that we use to describe the token
"""
# test version: input_tokens = torch.tensor([2, 3, 5, 1]) # dummy ids

with open("the-verdict.txt", "r", encoding="utf-8") as f:
	raw_text = f.read()

# max_length - # of tokens per tensor aka context
# stride - # of tokens we shift to the right before computing next tensor
# batch_size =:w # of tensors the dataloader returns from the dataset. The set returns tensors in pairs of input/target 
max_length = 4
dataloader = create_dataloader(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
input_tokens, target_tokens = next(data_iter)
print("input tensor tokens:\n", input_tokens.shape, "\n")
token_embeddings = voca_embedding_layer(input_tokens)
print("token_embeddings:\n", token_embeddings.shape, "\n")

"""
 the positional embedding layer is a matrix of random numbers that reprent the position of the token in the batch (position of the word in a sentence) the number of tokens in a single batch represent the context for that token (context_length)
"""
context_length = max_length # number of tokens in each batch (number of words in each sentence)
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print("postional_embeddings:\n", pos_embeddings.shape, "\n")


"""
 the input embeddings matrix is the one we provide to the LLM for training. we compute the input embedding matrix by combining the token embedding matrix with the positional embedding matrix to add position information for each token in each batch
"""
input_embeddings = token_embeddings + pos_embeddings
print("input_embeddings:\n", input_embeddings.shape, "\n")

