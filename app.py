import streamlit as st 
from rag_pipeline import prepare_pipeline  
# --- Pipeline Fonksiyonu (Sadece Bir Kez Çalışır) ---
@st.cache_resource
def load_pipeline(): 
     return prepare_pipeline() 
    # import time
    # time.sleep(1.0)  
    # def answer_question(query):
    #     if "aşı" in query.lower() and "ateş" in query.lower():
    #         return {"result": "**Aşı sonrası hafif ateş:** Vücudun aşıya tepkisinin normal bir işaretidir ve genellikle birkaç gün içinde geçer. Ancak yüksek veya uzun süren ateş için doktora danışılmalıdır.", "source": "Sağlık Bakanlığı Bilgi Notu"}
    #     else:
    #         return {"result": f"Bilgilendirme talebiniz: '{query}'. Lütfen daha spesifik olunuz. **Tıbbi teşhis yerine geçmez.**", "source": "Yapay Zeka"}
    # return answer_question  
# --- STREAMLIT ---
st.set_page_config(
    page_title="InfoMed Chatbot", 
    page_icon="🩺",
    layout="wide"
) 
st.title("🩺 InfoMed Chatbot - RAG Tabanlı Sağlık Bilgi Asistanı")  
# --- PIPELINE YÜKLEME ---
try:
    with st.spinner("Bilgi Sistemi hazırlanıyor... (ilk seferde biraz uzun sürebilir)"):
        qa = load_pipeline()
except Exception as e:
    st.error("Sistem yüklenirken kritik bir hata oluştu. Lütfen uygulamayı yeniden başlatın.") 
    st.stop() 
# --- KULLANICI GİRİŞİ  --- 
col_input, col_button = st.columns([4, 1])
button_key = "sor_button"  

with col_input:
    query = st.text_input(
        "Sağlıkla ilgili bir sorunuzu yazın:",
        placeholder="Örn: 'Aşıdan sonra ateş olması normal mi?'",
        key="user_query_input",
    )

with col_button:
    st.markdown("<br>", unsafe_allow_html=True) 
    if st.button("🔍 Sor", type="primary",    key=button_key):
        # Butona tıklandığında session state'e kaydet
        st.session_state.submit_clicked = True
        st.session_state.last_query = query # Sorguyu kaydet
  
if st.session_state.get('submit_clicked', False):
     
    st.session_state.submit_clicked = False
    
    current_query = st.session_state.last_query
    
    if not current_query.strip():
        st.warning("Lütfen bir soru yazın.")
    else: 
        with st.spinner("Yanıt hazırlanıyor..."):
            try:
                # QA fonksiyonunu çağır
                result = qa(current_query)
                answer = result.get("result", "Yanıt bulunamadı.")
                source = result.get("source", None)
                # Yanıtı kutuda göster
                st.success(answer)
            except Exception as e:
                # Hata durumunu göster
                st.error("Yanıt hazırlanırken beklenmedik bir sistem hatası oluştu. Lütfen biraz sonra tekrar deneyiniz.")
 
st.caption("Bilgilendirme amaçlıdır. Tıbbi teşhis yerine geçmez.Profesyonel tıbbi tavsiye almayı unutmayınız.")