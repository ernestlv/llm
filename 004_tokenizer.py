import re
text = "Hello, word. This, is a test."
tokens = re.split(r'([.,]|\s)', text)
print(tokens)
