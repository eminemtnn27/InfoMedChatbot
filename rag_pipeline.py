import os
from dotenv import load_dotenv
import google.generativeai as genai

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.language_models.llms import LLM
from langchain.chains import RetrievalQA

# --- Ortam değişkenleri ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("🚨 GOOGLE_API_KEY bulunamadı! Lütfen .env dosyasına ekleyin.")

genai.configure(api_key=api_key)

# --- Google Gemini LLM ---
class GeminiLLM(LLM):
    @property
    def _llm_type(self):
        return "gemini"

    def _call(self, prompt: str, stop=None):
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text

# --- Doküman yükleme ---
def load_documents(data_path="data/health_faqs.txt"):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Veri dosyası bulunamadı: {data_path}")
    loader = TextLoader(data_path, encoding="utf-8")
    return loader.load()

# --- Doküman bölme ---
def split_documents(docs, chunk_size=600, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

# --- Vektör veritabanı oluşturma ---
def create_or_load_vectorstore(texts, persist_directory="vectorstore"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(texts, embedding=embeddings, persist_directory=persist_directory)
    db.persist()
    return db

# --- QA zinciri oluşturma ---
def build_qa_chain(db):
    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = GeminiLLM()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa

# --- Ana pipeline ---
def prepare_pipeline():
    docs = load_documents()
    chunks = split_documents(docs)
    db = create_or_load_vectorstore(chunks)
    qa = build_qa_chain(db)
    return qa

if __name__ == "__main__":
    qa = prepare_pipeline()
    result = qa("Aşıdan sonra ateş olması normal mi?")
    print(result["result"])
