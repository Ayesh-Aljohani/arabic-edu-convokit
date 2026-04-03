"""Arabic mathematical term detection with fixed regex boundaries."""

import re
import logging

logger = logging.getLogger(__name__)

MATH_LEXICON = {
    "numbers_basic": [
        "صفر", "اثنان", "ثلاثه", "اربعه", "خمسه", "سته", "سبعه",
        "ثمانيه", "تسعه", "عشره", "عشرون", "ثلاثون", "اربعون", "خمسون",
        "ستون", "سبعون", "ثمانون", "تسعون", "مئه", "مائه", "الف", "مليون",
        "احد عشر", "اثنا عشر", "ثلاثه عشر", "اربعه عشر", "خمسه عشر",
        "سته عشر", "سبعه عشر", "ثمانيه عشر", "تسعه عشر",
    ],
    "arithmetic_operations": [
        "جمع", "طرح", "ضرب", "قسمه", "حاصل", "ناتج", "مجموع", "فرق",
        "باقي", "زائد", "ناقص", "مقسوم", "يساوي", "عمليه حسابيه",
        "حساب", "اضافه", "اضف", "اطرح",
    ],
    "fractions_decimals": [
        "كسر", "كسور", "بسط", "مقام", "نصف", "ثلث", "ربع", "خمس",
        "سدس", "سبع", "ثمن", "تسع", "عشر", "كسر عشري", "فاصله عشريه",
        "عدد عشري", "كسر اعتيادي", "كسر مركب", "اختصار", "تبسيط",
        "كسور متكافئه", "مقامات", "توحيد المقامات",
    ],
    "algebra": [
        "متغير", "معادله", "معادلات", "حل", "مجهول", "قيمه", "تعبير",
        "حد", "حدود", "معامل", "اس", "قوه", "جذر", "تربيع", "تكعيب",
        "متعدد الحدود", "كثير الحدود", "داله", "متباينه", "نسبه", "تناسب",
        "عامل", "عوامل", "تحليل", "صيغه",
    ],
    "geometry": [
        "مثلث", "مربع", "مستطيل", "دائره", "خط", "نقطه", "زاويه",
        "متوازي اضلاع", "شبه منحرف", "معين", "خماسي", "سداسي",
        "مضلع", "قطر", "نصف قطر", "محيط", "مساحه",
        "مكعب", "اسطوانه", "كره", "مخروط", "هرم", "منشور",
        "سطح", "وجه", "حافه", "راس", "رؤوس", "ضلع", "اضلاع",
        "قاعده", "ارتفاع", "عرض", "طول",
    ],
    "geometry_properties": [
        "متطابق", "متشابه", "متوازي", "متعامد", "عمودي", "افقي",
        "قائمه", "حاده", "منفرجه", "مستقيمه", "محور تماثل",
        "تماثل", "انعكاس", "دوران", "انتقال", "تحويل",
    ],
    "measurement": [
        "متر", "سنتيمتر", "ملليمتر", "كيلومتر", "بوصه", "قدم",
        "ميل", "يارده", "لتر", "ملليلتر", "غرام", "كيلوغرام",
        "طن", "ساعه", "اسبوع", "شهر",
        "سنه", "درجه", "وحده", "قياس",
    ],
    "statistics_probability": [
        "متوسط", "وسيط", "منوال", "مدى", "احتمال", "احتمالات",
        "بيانات", "رسم بياني", "جدول", "تكرار", "نسبه مئويه",
        "عينه", "مجتمع", "انحراف معياري", "تباين",
    ],
    "comparison_ordering": [
        "مساو", "ترتيب", "تصاعدي",
        "تنازلي", "تقريب", "تقدير",
    ],
    "patterns_relations": [
        "نمط", "انماط", "تسلسل", "متتاليه", "نظام", "قاعده",
        "ارتباط", "خاصيه", "خصائص", "برهان", "اثبات",
    ],
}


def _build_pattern(term: str) -> re.Pattern:
    """Build regex pattern using whitespace/boundary-aware matching for Arabic."""
    escaped = re.escape(term)
    return re.compile(r"(?:^|(?<=\s))" + escaped + r"(?:$|(?=\s))")


# Pre-compile all patterns
_COMPILED_PATTERNS: dict[str, list[tuple[str, re.Pattern]]] = {}
for _cat, _terms in MATH_LEXICON.items():
    _COMPILED_PATTERNS[_cat] = [(t, _build_pattern(t)) for t in _terms]


def count_math_terms(text: str) -> dict:
    """Count mathematical terms in Arabic text by category."""
    counts = {}
    total = 0
    for category, patterns in _COMPILED_PATTERNS.items():
        cat_count = 0
        for term, pattern in patterns:
            matches = pattern.findall(text)
            cat_count += len(matches)
        counts[category] = cat_count
        total += cat_count
    counts["total"] = total
    return counts


def has_math_content(text: str) -> bool:
    """Check if text contains any mathematical terms from the lexicon."""
    for patterns in _COMPILED_PATTERNS.values():
        for _, pattern in patterns:
            if pattern.search(text):
                return True
    return False


def calculate_math_density(text: str) -> float:
    """Calculate math content density as ratio of math terms to total words."""
    if not isinstance(text, str) or not text.strip():
        return 0.0
    words = text.split()
    if not words:
        return 0.0
    counts = count_math_terms(text)
    return counts["total"] / len(words)


def get_lexicon_size() -> int:
    """Return total number of terms in the math lexicon."""
    return sum(len(terms) for terms in MATH_LEXICON.values())
