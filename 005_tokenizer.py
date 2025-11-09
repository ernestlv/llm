import re
text = "Hello, word. This, is a test."
tokens = re.split(r'([.,]|\s)', text)
tokens = [item for item in tokens if item.strip()]
print(tokens)
