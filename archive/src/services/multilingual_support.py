"""
Multilingual AYUSH Terminology Support
Supports Hindi, Tamil, Arabic, Sanskrit and regional languages
"""

MULTILINGUAL_TRANSLATIONS = {
    "fever": {
        "english": "Fever",
        "hindi": "बुखार",
        "sanskrit": "ज्वर",
        "tamil": "காய்ச்சல்",
        "arabic": "حمى",
        "ayurveda_term": "Jwara",
        "siddha_term": "Kaycchal", 
        "unani_term": "Humma"
    },
    "diabetes": {
        "english": "Diabetes",
        "hindi": "मधुमेह",
        "sanskrit": "मधुमेह",
        "tamil": "நீரிழிவு",
        "arabic": "داء السكري",
        "ayurveda_term": "Madhumeha",
        "siddha_term": "Madhumeham",
        "unani_term": "Ziabetus"
    },
    "headache": {
        "english": "Headache",
        "hindi": "सिरदर्द",
        "sanskrit": "शिरशूल",
        "tamil": "தலைவலி",
        "arabic": "صداع",
        "ayurveda_term": "Shirashoola",
        "siddha_term": "Thalainokkadu",
        "unani_term": "Suda"
    },
    "joint_pain": {
        "english": "Joint Pain",
        "hindi": "जोड़ों का दर्द",
        "sanskrit": "संधिवात",
        "tamil": "மூட்டு வலி",
        "arabic": "ألم المفاصل",
        "ayurveda_term": "Sandhivata",
        "siddha_term": "Keel Vayu",
        "unani_term": "Waja-ul-Mafasil"
    }
}

def get_multilingual_term(condition: str, language: str = "english") -> str:
    """Get term in specified language"""
    condition_key = condition.lower().replace(" ", "_")
    if condition_key in MULTILINGUAL_TRANSLATIONS:
        return MULTILINGUAL_TRANSLATIONS[condition_key].get(language, condition)
    return condition

def get_all_languages_for_term(condition: str) -> dict:
    """Get term in all supported languages"""
    condition_key = condition.lower().replace(" ", "_")
    if condition_key in MULTILINGUAL_TRANSLATIONS:
        return MULTILINGUAL_TRANSLATIONS[condition_key]
    return {"english": condition}

if __name__ == "__main__":
    # Test multilingual support
    print(" Multilingual AYUSH Support Ready!")
    print("\n Sample translations for 'fever':")
    translations = get_all_languages_for_term("fever")
    for lang, term in translations.items():
        print(f"  {lang}: {term}")
