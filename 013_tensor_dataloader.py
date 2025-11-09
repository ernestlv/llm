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

with open("the-verdict.txt", "r", encoding="utf-8") as f:
	raw_text = f.read()

# max_length - # of tokens per tensor aka context
# stride - # of tokens we shift to the right before computing next tensor
# batch_size = # of tensors the dataloader returns from the dataset. The set returns tensors in pairs of input/target 
dataloader = create_dataloader(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("inputs:\n", inputs)
print("\ntargets:\n", targets)
