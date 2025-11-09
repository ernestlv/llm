# slide window example to train llm to predict next token
import re
import tiktoken   

with open("the-verdict.txt", "r", encoding="utf-8") as f:
   raw_text = f.read();

tokenizer = tiktoken.get_encoding("gpt2")
ids = tokenizer.encode(raw_text)
print(len(ids))
# remove first 50 tokens
enc_sample = ids[50:]

# number of tokens in the input
context_size = 4
#input
x = enc_sample[:context_size]
#target - slide one place to the right - contains token to be predicted by llm
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y:      {y}")
