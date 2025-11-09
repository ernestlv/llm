# slide window example to train llm to predict next token
import re
import tiktoken   

with open("the-verdict.txt", "r", encoding="utf-8") as f:
   raw_text = f.read();

tokenizer = tiktoken.get_encoding("gpt2")
ids = tokenizer.encode(raw_text)
# remove first 50 tokens
enc_sample = ids[50:]

# number of tokens in the input
context_size = 4
for i in range(1, context_size+1):
   context = enc_sample[:i]
   desired = enc_sample[i]
   print(tokenizer.decode(context), "--->", tokenizer.decode([desired]))
