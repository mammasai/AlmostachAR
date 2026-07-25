# واجهة المحادثة الرئيسية لـ AlmostachAR

import os
import sys
import uuid
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.orchestrator import run_chat_turn


st.set_page_config(
    page_title="AlmostachAR — المستشار",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stApp {
        direction: rtl;
    }
    .stApp h1 {
        text-align: center !important;
    }
    .intro-block,
    .intro-block p {
        direction: rtl;
        text-align: center !important;
    }
    .stChatMessage,
    .stChatMessage p,
    .stChatMessage li,
    .stChatMessage span {
        direction: rtl;
        text-align: right !important;
    }
    [data-testid="stSidebar"] {
        direction: rtl;
        text-align: center;
    }
    [data-testid="stChatInput"] textarea {
        display: flex;
        align-items: center;
        min-height: 44px;
        padding-top: 12px;
        padding-bottom: 12px;
    }
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarCustom"],
    [data-testid="stChatMessageAvatarAssistant"] {
        display: none !important;
    }
    [data-testid="stChatMessage"] {
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def new_conversation():
    conv_id = str(uuid.uuid4())
    st.session_state.conversations[conv_id] = {
        "title": "محادثة جديدة",
        "messages": [],
        "history": "",
    }
    st.session_state.current_conversation_id = conv_id
    return conv_id


if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "current_conversation_id" not in st.session_state:
    new_conversation()

if "uploaded_file_path" not in st.session_state:
    st.session_state.uploaded_file_path = ""

current = st.session_state.conversations[st.session_state.current_conversation_id]


with st.sidebar:
    st.title("AlmostachAR")
    st.caption("مستشارك الذكي لهندسة معالجة اللغة العربية الطبيعية")

    st.divider()

    st.subheader("رفع ملف بيانات")
    uploaded_file = st.file_uploader(
        "اختار ملف CSV, Excel, PDF, DOCX, DOC, أو TXT",
        type=["csv", "xlsx", "xls", "txt", "pdf", "docx", "doc"],
    )
    st.divider()
    if st.button("محادثة جديدة", use_container_width=True):
        new_conversation()
        st.rerun()

    st.divider()

    if uploaded_file is not None:
        raw_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "raw"
        )
        os.makedirs(raw_dir, exist_ok=True)
        save_path = os.path.join(raw_dir, uploaded_file.name)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state.uploaded_file_path = save_path
        st.success(f"تم رفع: {uploaded_file.name}")

    if st.session_state.uploaded_file_path:
        st.info(f"الملف الحالي:\n{os.path.basename(st.session_state.uploaded_file_path)}")
        if st.button("إزالة الملف الحالي"):
            st.session_state.uploaded_file_path = ""
            st.rerun()

    st.divider()
    st.subheader("المحادثات السابقة")

    for conv_id in reversed(list(st.session_state.conversations.keys())):
        conv = st.session_state.conversations[conv_id]
        is_current = conv_id == st.session_state.current_conversation_id
        label = ("◉ " if is_current else "") + conv["title"]
        if st.button(label, key=f"conv_{conv_id}", use_container_width=True):
            st.session_state.current_conversation_id = conv_id
            st.rerun()

st.title("AlmostachAR")

if not current["messages"]:
    st.markdown(
        """
        <div class="intro-block">
        أهلاً بيك، أنا <b>AlmostachAR</b>، مستشارك الذكي لبناء نماذج معالجة اللغة
        العربية الطبيعية
        <br><br>
        تقدر تسألني أي سؤال عام، أو ترفع ملف بيانات من الشريط الجانبي
        تطلب مني أفحصه، أنظفه، أقترح لك نموذج، أو أي شيء تحتاجه
        </div>
        """,
        unsafe_allow_html=True,
    )

for message in current["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_input = st.chat_input("اكتب هنا...")

if user_input:
    if current["title"] == "محادثة جديدة":
        current["title"] = user_input.strip()[:28] + ("…" if len(user_input.strip()) > 28 else "")

    current["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("جاري التفكير..."):
            history_lines = current["history"].strip().split("\n\n")
            trimmed_history = "\n\n".join(history_lines[-3:])

            try:
                response = run_chat_turn(
                    user_message=user_input,
                    conversation_history=trimmed_history,
                    file_path=st.session_state.uploaded_file_path,
                )
            except Exception as e:
                response = f"حدث خطأ: {str(e)}"

        st.markdown(response)

    current["messages"].append({"role": "assistant", "content": response})
    current["history"] += f"المستخدم: {user_input}\nAlmostachAR: {response}\n\n"