import chromadb
from app.models.models import SearchResult, SearchResultsList
import asyncio
from pathlib import Path


DATABASE_DIR = str(Path(__file__).parent / 'src' / 'database')
COLLECTION_NAME = 'music'


class Database:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=DATABASE_DIR)
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)

    # временный костыль, потом базу пересоздам адекватно
    def _convert_str_to_array(self, data_string):
        return data_string[1:-1].replace("'", '').split(', ')

    async def search(self, query_embedding):
        loop = asyncio.get_running_loop()
        search_results = await loop.run_in_executor(
            None,
            lambda: self.collection.query(
                query_embeddings=[query_embedding],
                n_results=5
            )
        )

        return SearchResultsList(
            search_results=[
                SearchResult(
                    song_name=result['name'],
                    artist_name=result['artist_name'],
                    album_name=result['album_name'],
                    releasedate=result['releasedate'],
                    vocalinstrumental=result['musicinfo.vocalinstrumental'],
                    artist_gender=result['musicinfo.gender'],
                    track_speed=result['musicinfo.speed'],
                    genres=self._convert_str_to_array(result['musicinfo.tags.genres']),
                    instruments=self._convert_str_to_array(result['musicinfo.tags.instruments']),
                    tags=self._convert_str_to_array(result['musicinfo.tags.vartags']),
                    confidence=1-distance
                )
                for distance, result in zip(search_results['distances'][0], search_results['metadatas'][0])
            ]
        )
