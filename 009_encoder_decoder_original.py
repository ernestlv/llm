import re

with open("the-verdict.txt", "r", encoding="utf-8") as f:
   raw_text = f.read()

tokens = re.split(r'([.,:;?_!"()\']|--|\s)', raw_text)
tokens = [item for item in tokens if item.strip()]
voca = sorted(list(set(tokens)))
voca.extend(["<|endoftext|>", "<|unk|>"])
voca = {token:i for i,token in enumerate(voca)}

class Tokenizer:
   
   def __init__(self, voca):
      self.str_to_int = voca
      self.int_to_str = {i:s for s,i in voca.items()}

   def encode(self, text):
      tokens = re.split(r'([.,:;?_!"()\']|--|<\|endoftext\|>|\s)', text)
      tokens = [item for item in tokens if item.strip()]
      tokens = [item if item in self.str_to_int else "<|unk|>" for item in tokens]
      ids = [self.str_to_int[s] for s in tokens]
      return ids

   def decode(self, ids):
      text = " ".join([self.int_to_str[i] for i in ids])
      text = re.sub(r'\s+([.,:;?_!"()\'])', r'\1', text)
      return text



tokenizer = Tokenizer(voca)
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = "<|endoftext|>".join((text1, text2))
print(text)
ids = tokenizer.encode(text)
print(ids)
text = tokenizer.decode(ids)
print(text)
