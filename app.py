import streamlit as st
from rag_pipeline import prepare_pipeline

st.set_page_config(page_title="InfoMedChatbot", page_icon="🩺")
st.title("🩺 InfoMedChatbot - Hasta Bilgilendirme Asistanı")
st.write("Bilgilendirme amaçlıdır. Tıbbi teşhis yerine geçmez.")

with st.spinner("Pipeline hazırlanıyor... (ilk seferde biraz uzun sürebilir)"):
    qa = prepare_pipeline()

query = st.text_input("Sağlıkla ilgili bir sorunuzu yazın:")
if st.button("Sor"):
    if not query.strip():
        st.warning("Lütfen bir soru yazın.")
    else:
        with st.spinner("Yanıt hazırlanıyor..."):
            try:
                result = qa(query)
                answer = result["result"]
                st.success(answer)
            except Exception as e:
                st.error(f"Hata: {e}")
