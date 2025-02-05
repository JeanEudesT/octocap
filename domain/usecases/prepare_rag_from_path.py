from domain.models.rag import Rag


class PrepareRagFromPath:
    _embedding_repository: any

    def __init__(self, persistance_repository):
        self._embedding_repository = persistance_repository

    def execute(self, path, ext):
        rag = Rag()
        rag.load_documents(path, ext)
        rag.create_chunks()
        embeddings_to_save = rag.create_embeddings()
        self._embedding_repository.save(embeddings_to_save)
