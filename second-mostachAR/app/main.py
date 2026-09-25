import os
import sys
import uuid
from datetime import datetime, timezone
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.orchestrator import run_chat_turn
from auth import sign_up, sign_in, sign_out, get_google_oauth_url, sign_in_with_google_code, get_client
from db import (
    fetch_conversations,
    upsert_conversation,
    delete_conversation,
    delete_all_conversations,
    rename_conversation,
)

try:
    from fpdf import FPDF
    PDF_EXPORT_AVAILABLE = True
except ImportError:
    PDF_EXPORT_AVAILABLE = False


LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo.png")

GOOGLE_G_SVG = """<svg width="18" height="18" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">
<path fill="#FFC107" d="M43.6 20.5H42V20H24v8h11.3C33.7 32.9 29.3 36 24 36c-6.6 0-12-5.4-12-12s5.4-12 12-12c3.1 0 5.8 1.1 8 3l6-6C34.5 5.1 29.6 3 24 3 12.4 3 3 12.4 3 24s9.4 21 21 21 21-9.4 21-21c0-1.4-.1-2.7-.4-3.5z"/>
<path fill="#FF3D00" d="M6.3 14.7l6.6 4.8C14.5 16 18.9 13 24 13c3.1 0 5.8 1.1 8 3l6-6C34.5 5.1 29.6 3 24 3 16.3 3 9.7 7.3 6.3 14.7z"/>
<path fill="#4CAF50" d="M24 45c5.2 0 10-2 13.6-5.2l-6.3-5.3C29.3 36.5 26.8 37.5 24 37.5c-5.2 0-9.6-3.5-11.2-8.3l-6.5 5C9.7 40.7 16.3 45 24 45z"/>
<path fill="#1976D2" d="M43.6 20.5H42V20H24v8h11.3c-.8 2.3-2.3 4.3-4.2 5.7l6.3 5.3C40.9 36 44 30.6 44 24c0-1.4-.1-2.7-.4-3.5z"/>
</svg>"""


if "theme" not in st.session_state:
    st.session_state.theme = "dark"

if st.session_state.theme == "light":
    _THEME_VARS_CSS = """
    <style>
    :root {
        --background-color: #FFFFFF;
        --secondary-background-color: #FFFFFF;
        --text-color: #172033;
        --primary-color: #173B63;
        --ink-deep: #FFFFFF;
        --ink-mid: #FFFFFF;
        --gold: #173B63;
        --gold-soft: #173B63;
        --teal: #173B63;
        --teal-soft: #234F7D;
        --text-warm: #172033;
        --hairline: rgba(23, 59, 99, 0.16);
    }
    .stApp,
    [data-testid="stAppViewContainer"],
    section.main,
    [data-testid="stSidebar"],
    [data-testid="stSidebarContent"] {
        background: #FFFFFF !important;
        background-image: none !important;
    }
    </style>
    """
else:
    _THEME_VARS_CSS = """
    <style>
    :root {
        --background-color: #0B1026;
        --secondary-background-color: #0B1026;
        --text-color: #EAF2FA;
        --primary-color: #4B78B8;
        --ink-deep: #0B1026;
        --ink-mid: #111936;
        --gold: #6EA0D6;
        --gold-soft: #82B0E4;
        --teal: #4B78B8;
        --teal-soft: #6EA0D6;
        --text-warm: #EAF2FA;
        --hairline: rgba(123, 185, 237, 0.20);
    }
    .stApp,
    [data-testid="stAppViewContainer"],
    section.main,
    [data-testid="stSidebar"],
    [data-testid="stSidebarContent"] {
        background: #0B1026 !important;
        background-image: radial-gradient(circle at 50% 0%, rgba(78, 139, 205, 0.08), transparent 42%) !important;
    }
    </style>
    """

st.markdown(_THEME_VARS_CSS, unsafe_allow_html=True)


st.set_page_config(
    page_title="AlmostachAR — المستشار",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Reem+Kufi:wght@400..700&family=IBM+Plex+Sans+Arabic:wght@300;400;500;600;700&display=swap');

    html, body, .stApp, [class*="css"] {
        font-family: 'IBM Plex Sans Arabic', 'Tahoma', sans-serif !important;
    }

    .stApp {
        direction: rtl;
    }
    .stApp h1 {
        text-align: center !important;
        font-family: 'Reem Kufi', 'IBM Plex Sans Arabic', sans-serif !important;
        font-weight: 600 !important;
        color: var(--gold-soft) !important;
        letter-spacing: 0.5px;
        margin-bottom: 0 !important;
        -webkit-text-fill-color: unset !important;
    }
    .title-rule {
        width: 84px;
        height: 3px;
        margin: 10px auto 26px auto;
        border-radius: 3px;
        background: linear-gradient(90deg, transparent, var(--gold), transparent);
    }

    .intro-block,
    .intro-block p {
        direction: rtl;
        text-align: center !important;
        font-size: 15.5px;
        color: color-mix(in srgb, var(--text-color) 82%, transparent);
        line-height: 1.9;
    }
    .stChatMessage,
    .stChatMessage p,
    .stChatMessage li,
    .stChatMessage span {
        direction: rtl;
        text-align: right !important;
        color: var(--text-color) !important;
    }
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarCustom"],
    [data-testid="stChatMessageAvatarAssistant"] {
        display: none !important;
    }
    [data-testid="stChatMessage"] {
        justify-content: center;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 4px 0 !important;
    }
    [data-testid="stChatMessageContent"] {
        background: transparent !important;
        border: none !important;
        border-radius: 18px !important;
        box-shadow: none !important;
        padding: 12px 16px !important;
        font-size: 15px !important;
        line-height: 1.9 !important;
        animation: msg-in 0.3s ease-out;
    }
    [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessageContent"] li,
    [data-testid="stChatMessageContent"] span {
        font-size: 15px !important;
        line-height: 1.9 !important;
    }
    .msg-meta {
        font-size: 12px;
        color: color-mix(in srgb, var(--text-color) 50%, transparent);
        margin: 2px 4px;
    }
    @keyframes msg-in {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @media (prefers-reduced-motion: reduce) {
        [data-testid="stChatMessageContent"] { animation: none; }
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    html, body, .stApp, [data-testid="stAppViewContainer"], section.main {
        scrollbar-width: auto;
        scrollbar-color: var(--teal-soft) transparent;
    }
    ::-webkit-scrollbar {
        width: 12px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: color-mix(in srgb, var(--teal-soft) 55%, transparent);
        border-radius: 8px;
        border: 3px solid transparent;
        background-clip: content-box;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--teal-soft);
        background-clip: content-box;
    }

    .stButton button,
    .stFormSubmitButton button,
    [data-testid="stChatInput"] button {
        transition: transform 0.12s ease, box-shadow 0.12s ease !important;
    }
    .stButton button:active,
    .stFormSubmitButton button:active,
    [data-testid="stChatInput"] button:active {
        transform: scale(0.93) !important;
    }
    .stFormSubmitButton button {
        background: linear-gradient(135deg, var(--teal-soft), var(--teal)) !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 600 !important;
    }
    .stFormSubmitButton button:hover {
        box-shadow: 0 4px 16px color-mix(in srgb, var(--teal) 45%, transparent) !important;
    }
    /* Re-theme native Streamlit widgets so they follow the light/dark toggle
       instead of staying on Streamlit's built-in dark styling (which made
       dropdowns/selects look stuck black with a gold/yellow accent even in
       light mode). */
    [data-baseweb="select"] > div,
    [data-baseweb="base-input"] {
        background: var(--secondary-background-color) !important;
        border-color: var(--hairline) !important;
        color: var(--text-color) !important;
    }
    [data-baseweb="select"] span,
    [data-baseweb="select"] div {
        color: var(--text-color) !important;
    }
    [data-baseweb="popover"] ul,
    [data-baseweb="menu"] {
        background: var(--secondary-background-color) !important;
    }
    [data-baseweb="popover"] li,
    [data-baseweb="menu"] li {
        background: var(--secondary-background-color) !important;
        color: var(--text-color) !important;
    }
    [data-baseweb="popover"] li:hover,
    [data-baseweb="menu"] li:hover,
    [aria-selected="true"] {
        background: color-mix(in srgb, var(--teal) 22%, var(--secondary-background-color)) !important;
        color: var(--text-color) !important;
    }
    [data-testid="stExpander"] {
        background: var(--secondary-background-color) !important;
        border: 1px solid var(--hairline) !important;
        border-radius: 12px !important;
    }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] p,
    [data-testid="stExpander"] span {
        color: var(--text-color) !important;
    }
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background: var(--teal-soft) !important;
        border-color: var(--teal-soft) !important;
    }
    .stSlider [data-baseweb="slider"] > div > div {
        background: var(--teal-soft) !important;
    }
    .stRadio label span,
    .stRadio p {
        color: var(--text-color) !important;
    }

    [data-testid="stDecoration"] { display: none; }
    header[data-testid="stHeader"] { background: transparent; height: 3rem; }
    [data-testid="stToolbarActions"] { visibility: hidden; }

    [data-testid="stSidebar"] {
        direction: rtl;
        background: var(--background-color) !important;
        border-left: 1px solid color-mix(in srgb, var(--text-warm) 10%, transparent);
        padding-inline: 4px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] {
        display: none !important;
    }
    [data-testid="stSidebar"] .stButton button {
        background: linear-gradient(160deg, color-mix(in srgb, var(--teal) 10%, var(--ink-mid)), var(--ink-mid)) !important;
        border: 1.5px solid color-mix(in srgb, var(--teal-soft) 45%, transparent) !important;
        border-radius: 12px !important;
        color: var(--text-color) !important;
        font-weight: 500 !important;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        border-color: var(--teal-soft) !important;
        box-shadow: 0 4px 14px color-mix(in srgb, var(--teal) 20%, transparent) !important;
    }
    .sidebar-user-email {
        font-size: 13px;
        color: var(--text-color);
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        direction: ltr;
        text-align: right;
    }
    .sidebar-user-status {
        font-size: 13px;
        color: color-mix(in srgb, var(--text-color) 70%, transparent);
        margin: 4px 2px 6px 2px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #4F8A6B;
        box-shadow: 0 0 6px #4F8A6B;
        display: inline-block;
    }
    .chat-header-row {
        display: flex;
        justify-content: center;
        margin-bottom: -8px;
    }
    .model-badge {
        font-size: 11px;
        font-weight: 600;
        color: var(--teal-soft);
        background: color-mix(in srgb, var(--teal) 14%, transparent);
        border: 1px solid color-mix(in srgb, var(--teal-soft) 35%, transparent);
        border-radius: 999px;
        padding: 3px 12px;
    }

    /* Sidebar brand: logo and title are intentionally separated */
    .sidebar-brand-logo {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        margin: 2px 0 5px 0 !important;
    }
    .sidebar-brand-logo img {
        width: 78px !important;
        height: 78px !important;
        max-width: 78px !important;
        object-fit: contain !important;
    }
    .sidebar-app-name {
        font-weight: 800;
        font-size: 18px;
        line-height: 1.2;
        color: var(--gold) !important;
        letter-spacing: 0.3px;
        margin: 0 0 13px 0 !important;
        padding: 0 !important;
        text-align: center !important;
        display: block !important;
    }

    .block-container {
        padding-top: 2rem !important;
        max-width: 900px !important;
        margin: 0 auto !important;
    }

    div[data-testid="stChatInput"] {
        margin-top: 8px;
        border-radius: 14px;
        transition: box-shadow 0.25s ease;
        background: var(--secondary-background-color) !important;
        border: 1px solid color-mix(in srgb, var(--text-warm) 12%, transparent) !important;
        overflow: hidden;
    }
    div[data-testid="stChatInput"] * {
        background-color: transparent !important;
    }
    div[data-testid="stChatInput"] textarea {
        color: var(--text-color) !important;
    }
    div[data-testid="stChatInput"] button {
        background: color-mix(in srgb, var(--teal) 85%, transparent) !important;
        border-radius: 10px !important;
    }
    div[data-testid="stChatInput"]:focus-within {
        box-shadow: 0 0 0 1.5px color-mix(in srgb, var(--gold) 45%, transparent), 0 0 24px color-mix(in srgb, var(--gold) 12%, transparent) !important;
    }

    .st-key-login_box {
        max-width: 420px;
        margin: 40px auto 0 auto;
        background: color-mix(in srgb, var(--ink-mid) 45%, transparent);
        border: 1px solid color-mix(in srgb, var(--gold) 15%, transparent);
        border-radius: 20px;
        padding: 32px 28px;
        backdrop-filter: blur(10px);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.35);
    }
    .panel-section-label {
        font-size: 12px;
        font-weight: 600;
        color: color-mix(in srgb, var(--text-color) 55%, transparent);
        margin: 4px 2px 8px 2px;
    }
    .or-divider {
        text-align: center;
        color: color-mix(in srgb, var(--text-warm) 40%, transparent);
        font-size: 13px;
        margin: 14px 0;
        position: relative;
    }
    .or-divider::before,
    .or-divider::after {
        content: "";
        position: absolute;
        top: 50%;
        width: 40%;
        height: 1px;
        background: linear-gradient(90deg, transparent, color-mix(in srgb, var(--gold) 35%, transparent), transparent);
    }
    .or-divider::before { right: 0; }
    .or-divider::after { left: 0; }

    .google-btn {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        width: 100%;
        padding: 10px 16px;
        background: var(--secondary-background-color);
        border: 1.5px solid color-mix(in srgb, var(--text-warm) 15%, transparent);
        border-radius: 10px;
        color: var(--text-color) !important;
        text-decoration: none !important;
        font-weight: 600;
        font-size: 15px;
        transition: box-shadow 0.2s ease, transform 0.12s ease;
        direction: rtl;
    }
    .google-btn:hover {
        box-shadow: 0 4px 14px color-mix(in srgb, var(--text-warm) 12%, transparent);
    }
    .google-btn:active {
        transform: scale(0.97);
    }

    @media (max-width: 640px) {
        .stApp h1 {
            font-size: 1.8rem !important;
        }
        [data-testid="stChatMessageContent"] {
            padding: 12px 14px !important;
            font-size: 15px !important;
        }
        .st-key-login_box {
            max-width: 94vw !important;
            padding: 20px 16px !important;
            margin: 16px auto 0 auto !important;
        }
        div[data-testid="stChatInput"] {
            font-size: 15px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None and "code" in st.query_params:
    _code = st.query_params["code"]
    st.query_params.clear()
    _success, _message, _user = sign_in_with_google_code(_code)
    if _success:
        st.session_state.user = _user
        _convs, _fetch_err = fetch_conversations(_user.id)
        st.session_state.conversations = _convs
        if _fetch_err:
            st.session_state.conversations_load_error = _fetch_err
        st.rerun()
    else:
        st.session_state.google_login_error = _message


def render_login_page():
    with st.container(key="login_box"):
        if os.path.exists(LOGO_PATH):
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                st.image(LOGO_PATH, use_container_width=True)
        else:
            st.markdown("<h1 style='text-align:center;'>AlmostachAR</h1>", unsafe_allow_html=True)

        st.markdown(
            """
            <p style="text-align:center; color: gray;">مستشار ذكي لهندسة معالجة اللغة العربية الطبيعية</p>
            """,
            unsafe_allow_html=True,
        )

        tab_login, tab_signup = st.tabs(["تسجيل الدخول", "إنشاء حساب"])

        if st.session_state.get("google_login_error"):
            st.error(st.session_state.google_login_error)
            del st.session_state["google_login_error"]

        with tab_login:
            with st.form("login_form"):
                email = st.text_input("البريد الإلكتروني", key="login_email")
                password = st.text_input("كلمة السر", type="password", key="login_password")
                submitted = st.form_submit_button("دخول", use_container_width=True)

                if submitted:
                    if not email or not password:
                        st.error("يرجى تعبئة جميع الحقول.")
                    else:
                        success, message, user = sign_in(email, password)
                        if success:
                            st.session_state.user = user
                            _convs, _fetch_err = fetch_conversations(user.id)
                            st.session_state.conversations = _convs
                            if _fetch_err:
                                st.session_state.conversations_load_error = _fetch_err
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)

            with st.expander("نسيت كلمة السر؟"):
                reset_email = st.text_input("البريد الإلكتروني", key="reset_email")
                if st.button("إرسال رابط استعادة كلمة السر", key="reset_btn", use_container_width=True):
                    if not reset_email:
                        st.error("يرجى إدخال البريد الإلكتروني.")
                    else:
                        try:
                            get_client().auth.reset_password_for_email(reset_email)
                            st.success("تم إرسال رابط استعادة كلمة السر إلى بريدك الإلكتروني.")
                        except Exception as e:
                            st.error(f"تعذّر إرسال الرابط: {str(e)}")

            st.markdown('<div class="or-divider">أو</div>', unsafe_allow_html=True)

            google_url = get_google_oauth_url(redirect_to="http://localhost:8501")
            st.markdown(
                f'<a href="{google_url}" class="google-btn">{GOOGLE_G_SVG}'
                f'<span>المتابعة باستخدام Google</span></a>',
                unsafe_allow_html=True,
            )

        with tab_signup:
            with st.form("signup_form"):
                new_email = st.text_input("البريد الإلكتروني", key="signup_email")
                new_password = st.text_input("كلمة السر", type="password", key="signup_password")
                confirm_password = st.text_input("تأكيد كلمة السر", type="password", key="signup_confirm")
                submitted = st.form_submit_button("إنشاء حساب", use_container_width=True)

                if submitted:
                    if not new_email or not new_password:
                        st.error("يرجى تعبئة جميع الحقول.")
                    elif new_password != confirm_password:
                        st.error("كلمة السر وتأكيدها غير متطابقين.")
                    elif len(new_password) < 6:
                        st.error("يجب أن تتكون كلمة السر من 6 أحرف على الأقل.")
                    else:
                        success, message, user = sign_up(new_email, new_password)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)


def _categorize_by_date(updated_at_str):
    if not updated_at_str:
        return "اليوم"
    try:
        dt = datetime.fromisoformat(str(updated_at_str).replace("Z", "+00:00"))
    except Exception:
        return "أقدم"
    conv_date = dt.date()
    today = datetime.now(timezone.utc).date()
    delta_days = (today - conv_date).days
    if delta_days <= 0:
        return "اليوم"
    elif delta_days == 1:
        return "الأمس"
    elif delta_days <= 7:
        return "الأسبوع الحالي"
    return "أقدم"


def _build_transcript_txt(conv: dict) -> str:
    lines = [f"{conv.get('title', 'محادثة')}", "=" * 40, ""]
    for msg in conv.get("messages", []):
        speaker = "المستخدم" if msg["role"] == "user" else "AlmostachAR"
        lines.append(f"{speaker}: {msg['content']}")
        lines.append("")
    return "\n".join(lines)


def _pdf_safe_text(text: str) -> str:
    """يحوّل أي نص لصيغة آمنة لخط Helvetica الافتراضي في fpdf2 (بدون خط
    عربي Unicode مضمّن)، بدل ما يطيح البرنامج بخطأ FPDFUnicodeEncodingException."""
    try:
        return text.encode("latin-1", "replace").decode("latin-1")
    except Exception:
        return "[نص غير قابل للعرض بهذا الخط]"


def _build_transcript_pdf(conv: dict) -> bytes:
    """
    يبني PDF بسيط لكامل المحادثة. ملاحظة: fpdf2 بدون خط عربي Unicode مضمّن
    (زي Amiri أو Cairo) ما يعرض العربي بشكل صحيح (اتجاه/تشكيل الحروف).
    للحصول على PDF عربي دقيق شكلاً، لازم إضافة ملف خط .ttf عربي حقيقي
    عبر pdf.add_font(...) — هذا أبسط تنفيذ ممكن كنقطة بداية.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 8, _pdf_safe_text(conv.get("title", "محادثة")))
    pdf.ln(4)
    for msg in conv.get("messages", []):
        speaker = "User" if msg["role"] == "user" else "AlmostachAR"
        text = f"{speaker}: {msg['content']}"
        pdf.multi_cell(0, 7, _pdf_safe_text(text))
        pdf.ln(2)
    return bytes(pdf.output(dest="S"))


def render_chat_app():
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
        if st.session_state.conversations:
            st.session_state.current_conversation_id = next(iter(st.session_state.conversations))
        else:
            new_conversation()

    if "uploaded_file_path" not in st.session_state:
        st.session_state.uploaded_file_path = ""

    current = st.session_state.conversations[st.session_state.current_conversation_id]

    if st.session_state.get("conversations_load_error"):
        st.error(st.session_state.conversations_load_error, icon=":material/error:")
        del st.session_state["conversations_load_error"]

    with st.sidebar:
        if os.path.exists(LOGO_PATH):
            st.markdown('<div class="sidebar-brand-logo">', unsafe_allow_html=True)
            st.image(LOGO_PATH)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<p class="sidebar-app-name"></p>', unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='text-align:center;'>المستشار</h3>", unsafe_allow_html=True)

        col_new, col_theme = st.columns([3, 1])
        with col_new:
            if st.button("محادثة جديدة", use_container_width=True, key="new_chat_btn", icon=":material/add:"):
                new_conversation()
                st.rerun()
        with col_theme:
            theme_icon = ":material/light_mode:" if st.session_state.theme == "dark" else ":material/dark_mode:"
            if st.button("", key="theme_toggle_btn", icon=theme_icon, help="تبديل المظهر"):
                st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
                st.rerun()

        st.divider()

        if "specialty_choice" not in st.session_state:
            st.session_state.specialty_choice = "عام (تلقائي)"
        specialty_options = [
            "عام (تلقائي)",
            "فحص جودة البيانات",
            "تنظيف النصوص العربية",
            "اختيار النموذج المناسب",
            "تدريب النماذج (Fine-tuning)",
            "تقييم الأداء",
            "اللهجات العربية المحكية",
        ]
        st.session_state.specialty_choice = st.selectbox(
            "تركيز المحادثة",
            specialty_options,
            index=specialty_options.index(st.session_state.specialty_choice),
            key="specialty_selectbox",
        )

        if "temperature" not in st.session_state:
            st.session_state.temperature = 0.4
        if "depth" not in st.session_state:
            st.session_state.depth = "مفصل"
        with st.expander("إعدادات الإجابة", icon=":material/tune:"):
            st.session_state.temperature = st.slider(
                "درجة الإبداع/التحديد", 0.0, 1.0, st.session_state.temperature, 0.1,
                key="temperature_slider",
            )
            st.session_state.depth = st.radio(
                "عمق الإجابة", ["موجز", "مفصل"],
                index=["موجز", "مفصل"].index(st.session_state.depth),
                key="depth_radio", horizontal=True,
            )

        with st.expander("تصدير وسجل الحوار", icon=":material/download:"):
            if not current["messages"]:
                st.caption("لا يوجد محتوى لتصديره بعد.")
            else:
                st.download_button(
                    "تنزيل كنص (TXT)",
                    data=_build_transcript_txt(current),
                    file_name=f"{current['title']}.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key="export_txt_btn",
                    icon=":material/description:",
                )
                if PDF_EXPORT_AVAILABLE:
                    try:
                        _pdf_bytes = _build_transcript_pdf(current)
                        st.download_button(
                            "تنزيل كـ PDF",
                            data=_pdf_bytes,
                            file_name=f"{current['title']}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            key="export_pdf_btn",
                            icon=":material/picture_as_pdf:",
                        )
                        st.caption("تصدير PDF الحالي أساسي وما يدعم عرض العربي بشكل مثالي بعد. TXT أدق حالياً.")
                    except Exception as _pdf_err:
                        st.caption(f"تعذّر توليد PDF حالياً ({str(_pdf_err)}). استخدمي TXT بدلاً منه.")
                else:
                    st.caption("تصدير PDF غير مفعّل. شغّلي: pip install fpdf2")

        if st.session_state.uploaded_file_path:
            st.info(f"الملف المرفق:\n{os.path.basename(st.session_state.uploaded_file_path)}")
            if st.button("إزالة الملف", key="remove_file_btn", icon=":material/close:"):
                st.session_state.uploaded_file_path = ""
                st.rerun()

        st.divider()

        if "renaming_id" not in st.session_state:
            st.session_state.renaming_id = None

        st.markdown('<p class="panel-section-label">المحادثات</p>', unsafe_allow_html=True)

        ordered_ids = list(reversed(list(st.session_state.conversations.keys())))
        buckets = {"اليوم": [], "الأمس": [], "الأسبوع الحالي": [], "أقدم": []}
        for conv_id in ordered_ids:
            conv = st.session_state.conversations[conv_id]
            buckets[_categorize_by_date(conv.get("updated_at"))].append(conv_id)

        for bucket_name in ["اليوم", "الأمس", "الأسبوع الحالي", "أقدم"]:
            conv_ids = buckets[bucket_name]
            if not conv_ids:
                continue
            st.caption(bucket_name)
            for conv_id in conv_ids:
                conv = st.session_state.conversations[conv_id]
                is_current = conv_id == st.session_state.current_conversation_id

                if st.session_state.renaming_id == conv_id:
                    new_title = st.text_input(
                        "الاسم الجديد", value=conv["title"],
                        key=f"rename_input_{conv_id}", label_visibility="collapsed",
                    )
                    confirm_col, cancel_col = st.columns(2)
                    with confirm_col:
                        if st.button("حفظ", key=f"confirm_rename_{conv_id}", use_container_width=True, icon=":material/check:"):
                            _ren_err = rename_conversation(st.session_state.user.id, conv_id, new_title)
                            if _ren_err:
                                st.session_state.conversations_load_error = _ren_err
                            else:
                                conv["title"] = new_title
                            st.session_state.renaming_id = None
                            st.rerun()
                    with cancel_col:
                        if st.button("إلغاء", key=f"cancel_rename_{conv_id}", use_container_width=True, icon=":material/close:"):
                            st.session_state.renaming_id = None
                            st.rerun()
                    continue

                label = conv["title"]
                if is_current:
                    st.markdown(
                        f"<style>.st-key-conv_{conv_id} button {{ "
                        f"background: linear-gradient(135deg, var(--teal-soft), var(--teal)) !important; "
                        f"color: #ffffff !important; border: none !important; font-weight: 700 !important; }}</style>",
                        unsafe_allow_html=True,
                    )
                row_col, ren_col, del_col = st.columns([4, 1, 1])
                with row_col:
                    if st.button(label, key=f"conv_{conv_id}", use_container_width=True):
                        st.session_state.current_conversation_id = conv_id
                        st.rerun()
                with ren_col:
                    if st.button("", key=f"ren_{conv_id}", help="إعادة تسمية", icon=":material/edit:"):
                        st.session_state.renaming_id = conv_id
                        st.rerun()
                with del_col:
                    if st.button("", key=f"del_{conv_id}", help="حذف المحادثة", icon=":material/delete:"):
                        _del_err = delete_conversation(st.session_state.user.id, conv_id)
                        if _del_err:
                            st.session_state.conversations_load_error = _del_err
                        del st.session_state.conversations[conv_id]

                        if conv_id == st.session_state.current_conversation_id:
                            if st.session_state.conversations:
                                st.session_state.current_conversation_id = next(
                                    iter(st.session_state.conversations)
                                )
                            else:
                                new_conversation()

                        st.rerun()

        st.divider()

        if "confirm_clear_all" not in st.session_state:
            st.session_state.confirm_clear_all = False

        if not st.session_state.confirm_clear_all:
            if st.button("مسح كل المحادثات", key="clear_all_btn", use_container_width=True, icon=":material/delete_sweep:"):
                st.session_state.confirm_clear_all = True
                st.rerun()
        else:
            st.warning("متأكد؟ هذا الإجراء يمسح كل المحادثات نهائياً.")
            yes_col, no_col = st.columns(2)
            with yes_col:
                if st.button("نعم، امسح الكل", key="confirm_clear_all_btn", use_container_width=True, icon=":material/check:"):
                    _clear_err = delete_all_conversations(st.session_state.user.id)
                    if _clear_err:
                        st.session_state.conversations_load_error = _clear_err
                    st.session_state.conversations = {}
                    new_conversation()
                    st.session_state.confirm_clear_all = False
                    st.rerun()
            with no_col:
                if st.button("إلغاء", key="cancel_clear_all_btn", use_container_width=True, icon=":material/close:"):
                    st.session_state.confirm_clear_all = False
                    st.rerun()

        st.divider()

        st.markdown(
            '<p class="sidebar-user-status"><span class="status-dot"></span> المستخدم النشط</p>',
            unsafe_allow_html=True,
        )
        with st.expander(st.session_state.user.email, icon=":material/person:"):
            if st.button("تسجيل الخروج", use_container_width=True, key="logout_btn", icon=":material/logout:"):
                sign_out()
                st.session_state.user = None
                st.session_state.conversations = {}
                st.rerun()

    with st.container(key="chat_area"):
        
        st.title("AlmostachAR")
        st.markdown('<div class="title-rule"></div>', unsafe_allow_html=True)

        if not current["messages"]:
            st.markdown(
                """
                <div class="intro-block">
                مرحباً، أنا <b>AlmostachAR</b>، مستشار ذكي لبناء نماذج معالجة اللغة
                العربية الطبيعية
                <br><br>
                يمكن طرح أي سؤال عام، أو إرفاق ملف بيانات من صندوق الكتابة
                وطلب فحصه، تنظيفه، اقتراح نموذج مناسب له، أو أي أمر آخر
                </div>
                """,
                unsafe_allow_html=True,
            )

        for _msg_idx, message in enumerate(current["messages"]):
            with st.container(key=f"msg_{_msg_idx}"):
                if message["role"] == "user":
                    st.markdown(
                        f"<style>"
                        f".st-key-msg_{_msg_idx} [data-testid='stChatMessage'] {{ "
                        f"direction: rtl !important; display: flex !important; "
                        f"flex-direction: row !important; justify-content: flex-start !important; }}"
                        f".st-key-msg_{_msg_idx} [data-testid='stChatMessageContent'] {{ "
                        f"background: #E8F3FE !important; "
                        f"border: none !important; max-width: 72% !important; color: #18324A !important; "
                        f"margin-inline-end: auto !important; margin-inline-start: 0 !important; "
                        f"border-radius: 18px 18px 5px 18px !important; "
                        f"padding: 12px 18px !important; box-shadow: none !important; }}"
                        f".st-key-msg_{_msg_idx} [data-testid='stChatMessageContent'] p, "
                        f".st-key-msg_{_msg_idx} [data-testid='stChatMessageContent'] li {{ "
                        f"color: #18324A !important; font-size: 15px !important; line-height: 1.9 !important; }}"
                        f"</style>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<style>"
                        f".st-key-msg_{_msg_idx} [data-testid='stChatMessage'] {{ "
                        f"direction: rtl !important; display: flex !important; "
                        f"flex-direction: row !important; justify-content: flex-end !important; }}"
                        f".st-key-msg_{_msg_idx} [data-testid='stChatMessageContent'] {{ "
                        f"background: transparent !important; border: none !important; "
                        f"max-width: 92% !important; margin-inline-start: auto !important; "
                        f"margin-inline-end: 0 !important; padding: 10px 16px !important; "
                        f"box-shadow: none !important; border-radius: 0 !important; }}"
                        f".st-key-msg_{_msg_idx} [data-testid='stChatMessageContent'] p, "
                        f".st-key-msg_{_msg_idx} [data-testid='stChatMessageContent'] li {{ "
                        f"color: var(--text-color) !important; font-size: 15px !important; line-height: 1.9 !important; }}"
                        f"</style>",
                        unsafe_allow_html=True,
                    )
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        prompt = st.chat_input(
            "اكتب رسالة أو أرفق ملفاً...",
            accept_file=True,
            file_type=["csv", "xlsx", "xls", "txt", "pdf", "docx", "doc", "png", "jpg", "jpeg"],
        )

        if prompt:
            user_text = prompt.text if hasattr(prompt, "text") else prompt["text"]
            uploaded_files = prompt.files if hasattr(prompt, "files") else prompt["files"]

            if uploaded_files:
                raw_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "raw"
                )
                os.makedirs(raw_dir, exist_ok=True)
                uploaded_file = uploaded_files[0]
                save_path = os.path.join(raw_dir, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.uploaded_file_path = save_path

            if not user_text and uploaded_files:
                user_text = f"تم إرفاق الملف: {uploaded_files[0].name}"

            if user_text:
                if current["title"] == "محادثة جديدة":
                    current["title"] = user_text.strip()[:28] + ("…" if len(user_text.strip()) > 28 else "")

                current["messages"].append({"role": "user", "content": user_text})

                history_lines = current["history"].strip().split("\n\n")
                trimmed_history = "\n\n".join(history_lines[-6:])

                _specialty_hint = (
                    "" if st.session_state.get("specialty_choice") == "عام (تلقائي)"
                    else st.session_state.get("specialty_choice", "")
                )

                try:
                    response = run_chat_turn(
                        user_message=user_text,
                        conversation_history=trimmed_history,
                        file_path=st.session_state.uploaded_file_path,
                        temperature=st.session_state.get("temperature", 0.4),
                        depth=st.session_state.get("depth", "مفصل"),
                        specialty_hint=_specialty_hint,
                    )
                except Exception as e:
                    response = f"حدث خطأ: {str(e)}"

                current["messages"].append({"role": "assistant", "content": response})
                current["history"] += f"المستخدم: {user_text}\nAlmostachAR: {response}\n\n"

                _save_err = upsert_conversation(
                    user_id=st.session_state.user.id,
                    conv_id=st.session_state.current_conversation_id,
                    title=current["title"],
                    messages=current["messages"],
                    history=current["history"],
                )
                if _save_err:
                    st.session_state.conversations_load_error = _save_err

                st.rerun()


if st.session_state.user is None:
    render_login_page()
else:
    render_chat_app()