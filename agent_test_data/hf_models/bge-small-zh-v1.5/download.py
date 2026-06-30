from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-zh-v1.5")
model = AutoModel.from_pretrained("BAAI/bge-small-zh-v1.5")
model.save_pretrained("./model")
tokenizer.save_pretrained("./tokenizer")