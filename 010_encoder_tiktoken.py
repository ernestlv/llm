import re
import tiktoken   

tokenizer = tiktoken.get_encoding("gpt2")
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of someunknownPlace."
text = "<|endoftext|>".join((text1, text2))
print(text)
ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
for i in ids:
 print(i, tokenizer.decode([i]))
print(ids)
text = tokenizer.decode(ids)
print(text)

