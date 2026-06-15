"""
AI-Powered AYUSH Assistant - Intelligent Features for Enhanced User Experience
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import json
import random
from typing import List, Dict
import datetime

# AI-powered symptom to treatment mapping
AYUSH_AI_DATABASE = {
    "symptoms": {
        "fever": {
            "ayurveda": {
                "term": "Jwara",
                "treatments": ["Tulsi", "Giloy", "Amla", "Turmeric"],
                "medicines": ["Mahasudarshan Churna", "Godanti Bhasma"],
                "lifestyle": ["Rest", "Light diet", "Warm water"]
            },
            "siddha": {
                "term": "Kaycchal",
                "treatments": ["Nilavembu", "Adathodai", "Tulasi"],
                "medicines": ["Nilavembu Kashayam", "Adathodai Chooranam"],
                "lifestyle": ["Avoid cold foods", "Drink warm water"]
            },
            "unani": {
                "term": "Humma",
                "treatments": ["Afsanteen", "Tukhm-e-Kasoos", "Gul-e-Surkh"],
                "medicines": ["Habb-e-Surfa", "Arq-e-Afsanteen"],
                "lifestyle": ["Light diet", "Adequate rest"]
            }
        },
        "diabetes": {
            "ayurveda": {
                "term": "Madhumeha",
                "treatments": ["Methi", "Karela", "Jamun", "Gudmar"],
                "medicines": ["Chandraprabha Vati", "Nishamalaki Churna"],
                "lifestyle": ["Regular exercise", "Controlled diet", "Yoga"]
            },
            "siddha": {
                "term": "Madhumeham",
                "treatments": ["Vengayam", "Pavakkai", "Neem"],
                "medicines": ["Aavaarai Panchang Churnam"],
                "lifestyle": ["Physical activity", "Dietary control"]
            },
            "unani": {
                "term": "Ziabetus",
                "treatments": ["Tukhm-e-Hulba", "Karela", "Jamun"],
                "medicines": ["Qurs Tabasheer", "Jawarish Jalinus"],
                "lifestyle": ["Exercise", "Diet management"]
            }
        },
        "headache": {
            "ayurveda": {
                "term": "Shirashoola",
                "treatments": ["Brahmi", "Shankhpushpi", "Jatamansi"],
                "medicines": ["Pathyadi Kadha", "Saraswatarishta"],
                "lifestyle": ["Meditation", "Adequate sleep", "Stress management"]
            },
            "siddha": {
                "term": "Thalainokkadu",
                "treatments": ["Brahmi", "Mandukaparni"],
                "medicines": ["Brahmi Ghritam"],
                "lifestyle": ["Rest", "Oil massage"]
            },
            "unani": {
                "term": "Suda",
                "treatments": ["Ustukhuddus", "Gul-e-Surkh"],
                "medicines": ["Habb-e-Suda", "Roghan-e-Mom"],
                "lifestyle": ["Rest", "Head massage"]
            }
        }
    }
}

def get_ai_recommendations(symptom: str, system: str = "all") -> Dict:
    """Get AI-powered treatment recommendations"""
    symptom_lower = symptom.lower()
    recommendations = {"found": False, "systems": []}
    
    for sym_key, sym_data in AYUSH_AI_DATABASE["symptoms"].items():
        if sym_key in symptom_lower or symptom_lower in sym_key:
            recommendations["found"] = True
            recommendations["symptom"] = sym_key.title()
            
            if system == "all":
                for sys_name, sys_data in sym_data.items():
                    recommendations["systems"].append({
                        "system": sys_name.title(),
                        "traditional_term": sys_data["term"],
                        "natural_treatments": sys_data["treatments"],
                        "medicines": sys_data["medicines"],
                        "lifestyle_recommendations": sys_data["lifestyle"]
                    })
            else:
                if system.lower() in sym_data:
                    sys_data = sym_data[system.lower()]
                    recommendations["systems"].append({
                        "system": system.title(),
                        "traditional_term": sys_data["term"],
                        "natural_treatments": sys_data["treatments"],
                        "medicines": sys_data["medicines"],
                        "lifestyle_recommendations": sys_data["lifestyle"]
                    })
            break
    
    return recommendations

def generate_health_insights() -> List[Dict]:
    """Generate daily health insights"""
    insights = [
        {
            "category": "Ayurveda Tip",
            "title": "Morning Routine",
            "content": "Start your day with warm water and lemon to balance Agni (digestive fire)",
            "icon": ""
        },
        {
            "category": "Siddha Wisdom",
            "title": "Seasonal Health",
            "content": "Adjust your diet according to seasons - cooling foods in summer, warming in winter",
            "icon": ""
        },
        {
            "category": "Unani Guidance",
            "title": "Mind-Body Balance",
            "content": "Maintain temperament balance through proper diet, exercise, and mental peace",
            "icon": ""
        },
        {
            "category": "Modern Integration",
            "title": "Digital Wellness",
            "content": "Use technology wisely - our FHIR system helps doctors give you better traditional care",
            "icon": ""
        }
    ]
    
    return random.sample(insights, 2)  # Return 2 random insights

if __name__ == "__main__":
    # Test the AI functions
    print(" AI-Powered AYUSH Assistant Ready!")
    print("\n Sample Recommendations for 'fever':")
    result = get_ai_recommendations("fever")
    print(json.dumps(result, indent=2))
