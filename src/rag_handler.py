import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_text_splitters import RecursiveCharacterTextSplitter

embeddings_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
VECTOR_STORE_PATH = "/app/vector_store" 

class RAGHandler:
    def __init__(self, knowledge_base_path: str = "/app/knowledge_base"):
        """
        Initialise le RAG Handler.
        
        Args:
            knowledge_base_path (str): Le chemin vers le dossier contenant les documents de connaissances (.md).
        """
        self.embeddings = embeddings_model
        self.vector_store = self._load_or_create_vector_store(knowledge_base_path)

    def _load_documents(self, path: str) -> list:
        """Charge les documents depuis un chemin de répertoire spécifié."""
        loader = DirectoryLoader(
            path,
            glob="**/*.md", 
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        print(f"Chargement des documents depuis : {path}")
        return loader.load()

    def _create_vector_store(self, knowledge_base_path: str) -> FAISS | None:
        """Crée et sauvegarde la base de données vectorielle à partir des documents."""
        documents = self._load_documents(knowledge_base_path)
        if not documents:
            print("Aucun document trouvé pour créer le vector store.")
            return None
        
        print(f"{len(documents)} documents chargés. Création des vecteurs...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(texts, self.embeddings)
        os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
        vector_store.save_local(VECTOR_STORE_PATH)
        print(f"Vector store créé et sauvegardé dans : {VECTOR_STORE_PATH}")
        return vector_store

    def _load_or_create_vector_store(self, knowledge_base_path: str) -> FAISS | None:
        """Charge le vector store s'il existe, sinon le crée."""
        if os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
            print(f"Chargement du vector store existant depuis : {VECTOR_STORE_PATH}")
            return FAISS.load_local(
                VECTOR_STORE_PATH, 
                embeddings=self.embeddings, 
                allow_dangerous_deserialization=True 
            )
        else:
            print("Aucun vector store trouvé. Création d'un nouveau...")
            return self._create_vector_store(knowledge_base_path)

    def get_relevant_feedback(self, query: str, k: int = 1) -> list[str]:
        """Recherche les k conseils les plus pertinents pour une requête."""
        if not self.vector_store:
            return []
        results = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in results]

if __name__ == '__main__':
    print("Initialisation du RAG Handler en mode test...")
    handler = RAGHandler(knowledge_base_path="/app/knowledge_base") 
    if handler.vector_store and hasattr(handler.vector_store, 'index'):
        print(f"Vector store chargé avec {handler.vector_store.index.ntotal} vecteurs.")
        
        test_query = "gestion du stress"
        feedback = handler.get_relevant_feedback(test_query, k=2)
        
        print(f"\nTest de recherche pour : '{test_query}'")
        if feedback:
            print("Feedback pertinent trouvé :")
            for f in feedback:
                print(f"- {f[:150]}...") # Affiche un aperçu
        else:
            print("Aucun feedback pertinent trouvé pour cette requête.")
    else:
        print("Le RAG Handler n'a pas pu être initialisé ou le vector store est vide.")