"""
طبقة الوصول لقاعدة البيانات (Supabase) لحفظ واسترجاع محادثات كل مستخدم.
تعتمد على نفس عميل Supabase المُهيّأ في auth.py (بحيث تبقى جلسة المستخدم
المسجّل نشطة، وهذا ضروري حتى تعمل سياسات RLS المبنية على auth.uid()).
"""

from datetime import datetime, timezone
from auth import get_client


def fetch_conversations(user_id: str):
    """يجلب كل محادثات المستخدم من قاعدة البيانات، مرتبة من الأحدث للأقدم.
    يرجع (dict المحادثات, رسالة خطأ أو None)."""
    client = get_client()
    try:
        result = (
            client.table("conversations")
            .select("*")
            .eq("user_id", user_id)
            .order("updated_at", desc=True)
            .execute()
        )
    except Exception as e:
        return {}, f"فشل جلب المحادثات من قاعدة البيانات: {str(e)}"

    conversations = {}
    for row in result.data or []:
        conversations[row["id"]] = {
            "title": row.get("title", "محادثة جديدة"),
            "messages": row.get("messages", []),
            "history": row.get("history", ""),
            "updated_at": row.get("updated_at"),
        }
    return conversations, None


def upsert_conversation(user_id: str, conv_id: str, title: str, messages: list, history: str):
    """يحفظ (ينشئ أو يحدّث) محادثة واحدة في قاعدة البيانات.
    يرجع رسالة خطأ (نص) لو فشل، أو None لو نجح الحفظ."""
    client = get_client()
    try:
        client.table("conversations").upsert(
            {
                "id": conv_id,
                "user_id": user_id,
                "title": title,
                "messages": messages,
                "history": history,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        ).execute()
        return None
    except Exception as e:
        return f"فشل حفظ المحادثة في قاعدة البيانات: {str(e)}"


def rename_conversation(user_id: str, conv_id: str, new_title: str):
    """يعيد تسمية محادثة معيّنة. يرجع رسالة خطأ لو فشل، أو None لو نجح."""
    client = get_client()
    try:
        client.table("conversations").update(
            {"title": new_title, "updated_at": datetime.now(timezone.utc).isoformat()}
        ).eq("id", conv_id).eq("user_id", user_id).execute()
        return None
    except Exception as e:
        return f"فشل تعديل اسم المحادثة: {str(e)}"


def delete_conversation(user_id: str, conv_id: str):
    """يحذف محادثة معيّنة. يرجع رسالة خطأ لو فشل، أو None لو نجح."""
    client = get_client()
    try:
        client.table("conversations").delete().eq("id", conv_id).eq("user_id", user_id).execute()
        return None
    except Exception as e:
        return f"فشل حذف المحادثة: {str(e)}"


def delete_all_conversations(user_id: str):
    """يحذف كل محادثات المستخدم دفعة وحدة. يرجع رسالة خطأ لو فشل، أو None لو نجح."""
    client = get_client()
    try:
        client.table("conversations").delete().eq("user_id", user_id).execute()
        return None
    except Exception as e:
        return f"فشل مسح كل المحادثات: {str(e)}"