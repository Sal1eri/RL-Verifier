from datasets import load_dataset
from experiments.common import load_aime,load_aime_chat_format

dataset = load_aime_chat_format()
print(type(dataset[0]))
print(type(dataset))
print(dataset[0])