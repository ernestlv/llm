import re
text = "Hello, word. is this-- a test?"
tokens = re.split(r'([.,:;?_!"()\']|--|\s)', text)
tokens = [item for item in tokens if item.strip()]
print(tokens)
