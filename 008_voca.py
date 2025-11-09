import re

with open("the-verdict.txt", "r", encoding="utf-8") as f:
   raw_text = f.read()

tokens = re.split(r'([.,:;?_!"()\']|--|\s)', raw_text)
tokens = [item for item in tokens if item.strip()]
voca = sorted(set(tokens))
voca = {token:i for i,token in enumerate(tokens)}
print(enumerate(voca))
