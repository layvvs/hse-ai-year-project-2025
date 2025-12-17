from app.vector_search.utils import Tags

from sentence_transformers import SentenceTransformer
import numpy as np
import asyncio
from pathlib import Path


TEXT_PADDING = 'unknown'
EMBEDDING_SIZE = 384


class TextEmbedder:
    def __init__(self):
        models_path = str(Path(__file__).parent.parent / 'src' / 'models')
        self.text_embedder = SentenceTransformer(models_path)

    def _tags_to_string(self, tags: Tags) -> str:
        genres = tags.get('genres', [TEXT_PADDING])
        instruments = tags.get('instruments', [TEXT_PADDING])
        vartags = tags.get('tags', [TEXT_PADDING])
        return (genres, instruments, vartags)

    async def process(self, tags: Tags):
        tags_to_process = self._tags_to_string(tags)
        embedding = []
        for tag_group in tags_to_process:
            tag_group_length = len(tag_group)
            loop = asyncio.get_running_loop()
            tag_group_embedding = await loop.run_in_executor(
                None,
                lambda: self.text_embedder.encode(tag_group, batch_size=tag_group_length)
            )
            embedding.append(np.mean(tag_group_embedding, axis=0))
        return np.array(embedding).reshape(EMBEDDING_SIZE * 3)
