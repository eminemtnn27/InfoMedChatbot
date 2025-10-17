# chatbot.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
import google.generativeai as genai

# ---- .env yükle ----
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY bulunamadı. .env dosyasını kontrol et.")

# ---- Google Gemini API ayarla ----
genai.configure(api_key=api_key)

# ---- Gemini LLM sınıfı ----
class GeminiLLM(LLM):
    @property
    def _llm_type(self):
        return "gemini"

    def _call(self, prompt: str, stop=None) -> str:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text

# ---- Dokümanları yükle ----
def load_documents(data_path="data/health_faqs.txt"):
    loader = TextLoader(data_path, encoding="utf-8")
    return loader.load()

# ---- Dokümanları parçala ----
def split_documents(docs, chunk_size=600, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

# ---- Vektör veritabanını oluştur veya yükle ----
def create_or_load_vectorstore(texts, persist_directory="vectorstore"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(texts, embedding=embeddings, persist_directory=persist_directory)
    db.persist()
    return db

# ---- QA zincirini oluştur ----
def build_qa_chain(db):
    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = GeminiLLM()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa

# ---- Pipeline hazırla ----
def prepare_pipeline():
    docs = load_documents()
    texts = split_documents(docs)
    db = create_or_load_vectorstore(texts)
    qa = build_qa_chain(db)
    return qa

# ---- Test ----
if __name__ == "__main__":
    qa = prepare_pipeline()
    query = "Aşıdan sonra ateş olması normal mi?"
    result = qa.invoke(query)
    print("=== CEVAP ===")
    print(result["result"])
