import streamlit as st
from rag_pipeline import prepare_pipeline

# --- Pipeline (sadece bir kez yuklenir) ---
@st.cache_resource
def load_pipeline():
    return prepare_pipeline()

# --- Sayfa ayarlar? ---
st.set_page_config(
    page_title="?? InfoMed Chatbot",
    page_icon="??",
    layout="wide"
)

st.title("?? InfoMed Chatbot - RAG Tabanl? Sa?l?k Bilgi Asistan?")

# --- Pipeline yukleme ---
try:
    with st.spinner("Bilgi sistemi haz?rlan?yor... (ilk yukleme uzun surebilir)"):
        qa = load_pipeline()
except Exception as e:
    st.error("?? Sistem yuklenirken hata olu?tu. Lutfen sayfay? yenileyin.")
    st.stop()

# --- Kullan?c? giri?i ---
user_input = st.text_input(
    "Sa?l?kla ilgili bir sorunuzu yaz?n:",
    placeholder="Orn: 'A??dan sonra ate? olmas? normal mi?'"
)

if st.button("?? Sor"):
    if not user_input.strip():
        st.warning("Lutfen bir soru yaz?n.")
    else:
        with st.spinner("Yan?t haz?rlan?yor..."):
            try:
                result = qa(user_input)
                answer = result.get("result", "Yan?t bulunamad?.")
                st.success(answer)
            except Exception as e:
                st.error("?? Yan?t haz?rlan?rken bir hata olu?tu. Lutfen tekrar deneyin.")

st.caption("?? Bu asistan bilgilendirme amacl?d?r. T?bbi te?his yerine gecmez.")
