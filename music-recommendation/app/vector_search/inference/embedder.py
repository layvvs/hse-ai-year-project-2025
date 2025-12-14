from sentence_transformers import SentenceTransformer


class TextEmbedder:
    def __init__(self):
        self.text_embedder = SentenceTransformer('./src/models')

    def process(self, tag: str):
        return self.text_embedder.encode(tag)
