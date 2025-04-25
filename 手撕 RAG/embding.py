from ollama import embeddings

response = embeddings(model='nomic-embed-text', prompt='hello')

print(response['embedding'])
print(type(response['embedding']))