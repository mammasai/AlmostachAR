import os
from dotenv import load_dotenv
from supabase import create_client, Client, ClientOptions

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError(
        "SUPABASE_URL أو SUPABASE_KEY غير موجودين في ملف .env"
    )

# flow_type="pkce" ضروري لأي OAuth (زي جوجل) في تطبيق سيرفر-سايد زي Streamlit؛
# بدونه الـ tokens ترجع في الـ URL fragment (#...) اللي بايثون ما يقدر يقرأه.
_supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY,
    options=ClientOptions(flow_type="pkce"),
)


def sign_up(email: str, password: str):
    try:
        result = _supabase.auth.sign_up({"email": email, "password": password})
        if result.user:
            return True, "تم إنشاء الحساب بنجاح. تحققي من بريدك الإلكتروني لتأكيد الحساب.", result.user
        return False, "حدث خطأ غير متوقع أثناء إنشاء الحساب.", None
    except Exception as e:
        return False, f"فشل إنشاء الحساب: {str(e)}", None


def sign_in(email: str, password: str):
    try:
        result = _supabase.auth.sign_in_with_password({"email": email, "password": password})
        if result.user:
            return True, "تم تسجيل الدخول بنجاح.", result.user
        return False, "بيانات الدخول غير صحيحة.", None
    except Exception as e:
        return False, f"فشل تسجيل الدخول: {str(e)}", None


def get_google_oauth_url(redirect_to: str) -> str:
    """
    يرجع رابط صفحة تسجيل الدخول بجوجل. المستخدم لازم يضغط عليه (ما نقدر
    نسوي redirect تلقائي من بايثون في Streamlit)، وبعد ما يوافق بجوجل
    بيرجع تلقائياً لـ redirect_to مع ?code=... بالرابط.
    """
    result = _supabase.auth.sign_in_with_oauth({
        "provider": "google",
        "options": {"redirect_to": redirect_to},
    })
    return result.url


def sign_in_with_google_code(code: str):
    """
    تكمل عملية تسجيل الدخول بجوجل: تبادل الـ code اللي رجع بالرابط
    (?code=...) بجلسة مستخدم فعلية.
    """
    try:
        result = _supabase.auth.exchange_code_for_session({"auth_code": code})
        if result and result.user:
            return True, "تم تسجيل الدخول بجوجل بنجاح.", result.user
        return False, "تعذر إكمال تسجيل الدخول بجوجل.", None
    except Exception as e:
        return False, f"فشل تسجيل الدخول بجوجل: {str(e)}", None


def sign_out():
    try:
        _supabase.auth.sign_out()
    except Exception:
        pass


def get_client() -> Client:
    return _supabase