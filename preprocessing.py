import re
import emoji
from pyarabic.araby import strip_tatweel, strip_tashkeel, normalize_ligature, normalize_hamza

import re
import emoji
from pyarabic.araby import strip_tatweel, strip_tashkeel, normalize_ligature, normalize_hamza

# ============================================
# 1. Normalize Arabic Letters
# ============================================
def normalize_text(text):
    text = normalize_ligature(text)
    text = normalize_hamza(text)
    text = strip_tashkeel(text)
    text = strip_tatweel(text)

    replacements = {
        r'[إأآٱ]': 'ا',
        r'ى': 'ي',
        r'ة': 'ه',
        r'گ': 'ك',
        r'[ڤڨ]': 'ف',
        r'چ': 'ج',
        r'پ': 'ب'
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)
    return text

# ============================================
# 2. Handle Negation
# ============================================
negation_patterns = r'\b(مش|ما|لا|غير|مو|مفيش|محدش|ليس|ماعند|ماعنديش)\b'
def handle_negation(text):

    return re.sub(negation_patterns, 'لا', text)

# ============================================
# 4. Emoji Handling (Arabic Descriptions)
# ============================================
emoji_arabic_map = {
    # Hearts
    "❤️": "حب", "🧡": "حب", "💛": "حب", "💚": "حب", "💙": "حب",
    "💜": "حب", "🖤": "حب", "🤍": "حب", "🤎": "حب",

    # Face Expressions
    "😊": " مبتسم", "😂": "يضحك ", "😍": "معجب ", "😢": " حزين",
    "😭": " يبكي", "😡": "غاضب ", "😅": "احراج ", "😎": " رائع",
    "😒": " ملل", "🤔": "يفكر ", "🥰": "حب ", "🤗": "حضن ", "🙄": " تجاهل",
    "🥺": " رجاء", "😞": " محبط", "😩": " متعب", "😤": " منفعل",

    # Hands & Gestures
    "👍": "اعجاب", "👎": "عدم اعجاب", "🙏": "دعاء", "🤲": "رجاء",
    "✌️": "سلام", "💪": "قوة",

    # Nature & Objects
    "🌹": "وردة", "🔥": "نار", "🌟": "نجمة", "⭐": "نجمة", "🎉": "احتفال",
    "✨": "بريق", "🌞": "شمس", "🌙": "قمر"
}

def handle_emojis(text, emoji_map=emoji_arabic_map):
    def replace_known_emojis(char):
        return emoji_map.get(char, '')
    return ''.join(replace_known_emojis(c) if c in emoji.EMOJI_DATA else c for c in text)

# ============================================
# 5. Detect Non-Arabic
# ============================================
def is_arabic_text(text):
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    return arabic_chars / max(1, len(text)) > 0.4

# ============================================
# 6. Numbers & Punctuation
# ============================================

def remove_punctuation(text):
    return re.sub(r'[!"#$%&\'()*+,\-./:;<=>?@[\\]^_`{|}~،؛؟«»ـ]', ' ', text)

def handle_numbers(text, mode='convert'):
    text = re.sub(r'\d+', lambda m: m.group().translate(str.maketrans('0123456789', '٠١٢٣٤٥٦٧٨٩')), text)

    if mode == 'verbalize':
        num_map = {'٠':'صفر','١':'واحد','٢':'اتنين','٣':'تلاتة',
                   '٤':'اربعة','٥':'خمسة','٦':'ستة','٧':'سبعة',
                   '٨':'تمانية','٩':'تسعة'}
        for digit, word in num_map.items():
            text = text.replace(digit, word)
    return text

def preprocess(text):
    text = handle_emojis(text)
    if not is_arabic_text(text):
        return None
    text = normalize_text(text)
    text = handle_negation(text)
    text = remove_punctuation(text)
    text = handle_numbers(text, mode='remove')
    return text