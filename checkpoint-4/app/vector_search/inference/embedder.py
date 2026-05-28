from app.models.models import ForwardRequest

from sentence_transformers import SentenceTransformer
import numpy as np
import asyncio
from pathlib import Path


EMBEDDING_SIZE = 384


class TextEmbedder:
    def __init__(self):
        models_path = str(Path(__file__).parent.parent / 'src' / 'models')
        self.text_embedder = SentenceTransformer(models_path)

    async def process(self, tags: ForwardRequest):
        embedding = []
        for tag_group in tags.get_all_tags():
            tag_group_length = len(tag_group)
            loop = asyncio.get_running_loop()
            tag_group_embedding = await loop.run_in_executor(
                None,
                lambda: self.text_embedder.encode(tag_group, batch_size=tag_group_length)
            )
            embedding.append(np.mean(tag_group_embedding, axis=0))
        return np.array(embedding).reshape(EMBEDDING_SIZE * 3)
