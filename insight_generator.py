from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("API_KEY")
client = genai.Client(api_key=API_KEY)

def generate_text(prompt: str) -> str:
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return response.text


def generate_positive_points(reviews: list[str]) -> str:
    if not reviews:
        return "(لا توجد مراجعات إيجابية لتحليلها)"

    reviews_text = "\n".join([f"- {r}" for r in reviews])

    prompt = f"""
أنت صاحب مشروع تجاري وتقرأ المراجعات التالية والتي توصف بأنها إيجابية.

مهمتك هي استخراج **أقصى حد ٤ نقاط** مميزة من هذه المراجعات فقط. 
- لا تستنتج أو تفترض أي شيء من عندك.
- استخرج النقاط فقط من الكلمات والمعلومات الموجودة حرفيًا في المراجعات.
- إذا لم تستطع استخراج ٤ نقاط من المراجعات، اكتب فقط ما هو متاح.
- صحح الأخطاء الإملائية إن وجدت و صحح الكلمات المكتوبة بشكل خاطئ
- بداية كل نقطة يجب أن تكون - (شرطة ناقص) متبوعة بمسافة.

ملاحظات هامة:
- لا تكرر المعاني أو النقاط التي تشير إلى نفس المفهوم.
- لا تذكر كلمات بذيئة أو مهينة حتى لو وُجدت في المراجعات.

المراجعات:
{reviews_text}
"""
    output = generate_text(prompt)
    return output


def generate_negative_points(reviews: list[str]) -> str:
    if not reviews:
        return "(لا توجد مراجعات سلبية لتحليلها)"

    reviews_text = "\n".join([f"- {r}" for r in reviews])

    prompt = f"""
أنت صاحب مشروع تجاري وتقرأ المراجعات التالية والتي توصف بأنها سلبية.

مهمتك هي استخراج **أقصى حد ٤ نقاط** مميزة من هذه المراجعات فقط. 
- لا تستنتج أو تفترض أي شيء من عندك.
- استخرج النقاط فقط من الكلمات والمعلومات الموجودة حرفيًا في المراجعات.
- إذا لم تستطع استخراج ٤ نقاط من المراجعات، اكتب فقط ما هو متاح.
- صحح الأخطاء الإملائية إن وجدت و صحح الكلمات المكتوبة بشكل خاطئ
- بداية كل نقطة يجب أن تكون - (شرطة ناقص) متبوعة بمسافة.

ملاحظات هامة:
- لا تكرر المعاني أو النقاط التي تشير إلى نفس المفهوم.
- لا تذكر كلمات بذيئة أو مهينة حتى لو وُجدت في المراجعات.

المراجعات:
{reviews_text}
"""
    output = generate_text(prompt)
    return output
