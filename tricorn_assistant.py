from dotenv import load_dotenv
import streamlit as st
import os
from google import genai
from google.genai.errors import APIError
from io import BytesIO
# Import LangChain components
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import tempfile
import datetime
load_dotenv()

# Tentukan tanggal takedown yang Anda inginkan
# GANTI SESUAI TANGGAL 7 HARI KE DEPAN
TAKEDOWN_DATE = datetime.date(2025, 10, 10)

if datetime.date.today() >= TAKEDOWN_DATE:
    st.error("🛑 Aplikasi ini telah melewati masa uji coba dan tidak lagi aktif. Mohon hubungi pemilik proyek.")
    # Anda dapat menghentikan eksekusi logic LLM di sini, atau:
    # st.stop() # Menghentikan seluruh aplikasi

# Inisialisasi Klien Gemini
try:
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    print("Gemini Client berhasil diinisialisasi.")
except Exception as e:
    print(f"Error in initializing Gemini Client: {e}")
# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Tri Dharma Assistant (TriCorn Assistant) 🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Judul Utama Aplikasi
st.title("🎓 Tri Dharma Assistant (TriCorn Assistant)")
st.caption("Asisten AI Akademik yang didukung oleh Google Gemini dan RAG untuk menjawab pertanyaan berdasarkan dokumen institusi Anda.")

# --- Pengaturan Kunci API dan Model ---
# Kunci API akan dibaca dari Streamlit Secrets atau Environment Variable
# Pastikan Anda telah menetapkan GEMINI_API_KEY di Streamlit Secrets
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("⚠️ GEMINI_API_KEY tidak ditemukan. Harap atur di Streamlit Secrets atau Environment Variable.")
    st.stop()

# Inisialisasi Model dan Embeddings


@st.cache_resource(show_spinner=False)
def initialize_llm_components():
    # 1. LLM (Gemini 2.5 Flash untuk kecepatan dan performa)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.1
    )

    # 2. Embeddings (Untuk mengubah teks menjadi vektor)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",  # Model embedding terbaru
        google_api_key=GEMINI_API_KEY
    )

    return llm, embeddings


llm, embeddings = initialize_llm_components()

# --- Fungsi RAG (Core Logic) ---


# --- Fungsi RAG (Core Logic) ---
def get_rag_chain(docs, embeddings, llm):
    # 1. Split Dokumen
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(docs)

    # 2. Simpan ke Vector Store (Chroma DB)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )

    # 3. Inisialisasi Memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        # Menggunakan format string untuk kompatibilitas LLM yang lebih baik
        output_key='answer'
    )

    # --- PERBAIKAN UTAMA: MEMBUAT PROMPT TEMPLATE DENGAN SYSTEM INSTRUCTION ---

    # Dapatkan System Instruction (baik default atau kustom dari sidebar)
    # Ini memastikan kustomisasi persona di sidebar tetap berfungsi
    system_instruction_text = st.session_state.get("custom_prompt_text")

    # Definisi Default System Instruction
    if not system_instruction_text:
        system_instruction_text = (
            "Anda adalah Tri Dharma Assistant (TriCorn Assistant), asisten AI akademik yang formal, profesional, dan informatif. "
            "Fokus utama Anda adalah menjawab pertanyaan berdasarkan konteks dokumen yang disediakan (Tri Dharma: Pendidikan, Penelitian, Pengabdian). "
            "Selalu berikan jawaban yang akurat, merujuk pada dokumen jika memungkinkan. Jika informasi tidak ada di dokumen, katakan demikian dan berikan jawaban umum yang relevan secara formal."
        )

    # Template gabungan yang memasukkan instruction, konteks, dan pertanyaan
    RAG_PROMPT_TEMPLATE = (
        system_instruction_text +
        "\n\n### KONTEKS DOKUMEN YANG DISEDIAKAN UNTUK DIANALISIS ###\n---{context}---\n\n"
        "Sebagai Tri Dharma Assistant, berdasarkan konteks dokumen di atas, jelaskan dan jawab pertanyaan ini dengan profesional dan informatif. JIKA KONTEKS KOSONG ATAU TIDAK RELEVAN, katakan bahwa Anda tidak dapat menemukan informasi tersebut dalam dokumen, JANGAN katakan Anda tidak bisa mengakses file."
        "Pertanyaan: {question}"
    )

    # Buat objek PromptTemplate yang dibutuhkan oleh LangChain
    RAG_PROMPT = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        # Input yang dibutuhkan oleh prompt RAG
        input_variables=["context", "question"],
    )
    # --- AKHIR PERBAIKAN PROMPT ---

    # 4. Inisialisasi RAG Chain
    # Meningkatkan 'k' (jumlah dokumen yang diambil) menjadi 8 agar LLM bisa merangkum dokumen besar
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 8}),
        memory=memory,
        # Meneruskan PromptTemplate object
        combine_docs_chain_kwargs={
            "prompt": RAG_PROMPT
        }
    )

    st.session_state.rag_chain = rag_chain
    st.success("✅ Dokumen berhasil diproses! Chatbot siap digunakan.")


# --- Sidebar dan Input Dokumen ---
with st.sidebar:
    st.header("Upload Dokumen Institusi")

    # PERUBAHAN UTAMA: Menerima banyak file
    uploaded_files = st.file_uploader(
        "Upload file PDF (Panduan Skripsi, SOP Penelitian, dll.)",
        type="pdf",
        accept_multiple_files=True
    )

    # Proses dokumen setelah diunggah
    # Periksa apakah ada file yang diunggah
    if uploaded_files and "rag_chain" not in st.session_state:
        st.info(
            f"⏳ Sedang memproses {len(uploaded_files)} dokumen... Harap tunggu sebentar.")

        # Inisialisasi daftar kosong untuk menampung semua dokumen dari semua file
        all_docs = []

        try:
            # Loop melalui setiap file yang diunggah
            for uploaded_file in uploaded_files:
                # 1. Buat file sementara (Temporary File)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    # Tulis konten file yang diupload ke file sementara
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                # 2. Gunakan PyPDFLoader dengan path file sementara
                loader = PyPDFLoader(tmp_file_path)

                # Tambahkan dokumen yang dimuat ke daftar gabungan
                all_docs.extend(loader.load())

                # 3. Hapus file sementara setelah dimuat
                os.remove(tmp_file_path)

            # Membangun RAG Chain hanya SEKALI dengan semua dokumen yang digabungkan
            if all_docs:
                get_rag_chain(all_docs, embeddings, llm)

        except Exception as e:
            st.error(f"Gagal memproses dokumen: {e}")

    elif "rag_chain" in st.session_state:
        st.success(
            f"✅ Dokumen siap digunakan. Total dokumen: {len(st.session_state.rag_chain.retriever.vectorstore.get().get('ids', []))} chunks.")

    st.markdown("---")
    st.header("Parameters Kreatif")
    # Contoh parameter kustom: Custom System Prompt
    new_system_prompt = st.text_area(
        "Custom System Instruction/Persona (Optional):",
        value=st.session_state.get(
            "custom_prompt_text", "Anda adalah Tri Dharma Assistant, asisten AI akademik yang formal. Jawab pertanyaan berdasarkan dokumen yang diunggah. Jika tidak ada, berikan jawaban umum yang relevan secara profesional."),
        height=150
    )

    if st.button("Ubah Persona"):
        st.session_state.custom_prompt_text = new_system_prompt
        st.session_state.custom_prompt = new_system_prompt
        st.toast("Persona Chatbot berhasil diperbarui!")

    st.markdown("---")
    st.write("Aplikasi ini dibuat sebagai Final Project LLM-based Tools & Gemini API Integration by Hacktiv8 oleh I Made Satria Bimantara.")

# --- Main Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Pesan sambutan
    st.session_state.messages.append(
        {"role": "assistant", "content": "Selamat datang di Tri Dharma Assistant! Silakan upload dokumen PDF Anda di sidebar untuk memulai tanya jawab spesifik. Anda juga bisa langsung bertanya secara umum."})

# Tampilkan riwayat chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input dari pengguna
if prompt := st.chat_input("Tanyakan sesuatu tentang Tri Dharma atau dokumen yang telah diupload..."):
    # Tambahkan prompt pengguna ke riwayat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Proses respons dari LLM
    with st.chat_message("assistant"):
        with st.spinner("TriCorn Assistant sedang berpikir..."):

            # Logika RAG vs General Chat
            if "rag_chain" in st.session_state:
                # Gunakan RAG Chain jika dokumen telah diupload
                response = st.session_state.rag_chain({"question": prompt})
                ai_response = response["answer"]
            else:
                # Gunakan General Chat jika belum ada dokumen (Chat Langsung ke Gemini)
                # Tambahkan system instruction saat chat langsung
                general_prompt = st.session_state.get(
                    "custom_prompt_text", "Anda adalah Tri Dharma Assistant, asisten AI akademik yang formal. Jawab pertanyaan secara umum dan profesional.")

                # Menggunakan ChatGoogleGenerativeAI untuk chat biasa tanpa RAG
                general_llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=GEMINI_API_KEY,
                    temperature=0.1
                )

                # Konversi riwayat chat ke format yang bisa dipahami LLM (gemini-2.5-flash)
                history = [
                    (msg["role"], msg["content"])
                    for msg in st.session_state.messages[:-1]
                ]

                # Panggil LLM dengan history
                response_obj = general_llm.invoke(
                    input=prompt,
                    history=history,
                    system_instruction=general_prompt
                )
                ai_response = response_obj.content

            st.markdown(ai_response)

    # Tambahkan respons AI ke riwayat
    st.session_state.messages.append(
        {"role": "assistant", "content": ai_response})
