"""
Telemedicine Integration for AYUSH Practitioners
ABHA Integration and Virtual Consultation Support
"""

from datetime import datetime, timedelta
import uuid
import json

# Sample ABHA (Ayushman Bharat Health Account) integration
class ABHAIntegration:
    def __init__(self):
        self.base_url = "https://healthid.ndhm.gov.in"
        self.sessions = {}
    
    def verify_abha_id(self, abha_id: str) -> dict:
        """Verify ABHA ID and get patient details"""
        # Mock ABHA verification
        sample_patient = {
            "abha_id": abha_id,
            "name": "Sample Patient",
            "age": 35,
            "gender": "M",
            "constitution": "Vata-Pitta",
            "previous_consultations": 3,
            "preferred_system": "Ayurveda",
            "verified": True
        }
        return sample_patient
    
    def create_consultation_session(self, patient_abha: str, doctor_id: str) -> str:
        """Create virtual consultation session"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "patient_abha": patient_abha,
            "doctor_id": doctor_id,
            "start_time": datetime.now(),
            "status": "active",
            "consultation_type": "ayush_telemedicine"
        }
        return session_id

# Virtual Consultation Platform
class VirtualConsultation:
    def __init__(self):
        self.active_sessions = {}
        self.consultation_history = []
    
    def start_consultation(self, patient_data: dict, symptom_analysis: dict) -> dict:
        """Start virtual consultation with AI-assisted diagnosis"""
        consultation_id = str(uuid.uuid4())
        
        consultation = {
            "id": consultation_id,
            "patient": patient_data,
            "ai_analysis": symptom_analysis,
            "start_time": datetime.now().isoformat(),
            "status": "in_progress",
            "recommended_practitioners": self.find_suitable_practitioners(patient_data, symptom_analysis),
            "prescription_template": self.generate_prescription_template(symptom_analysis)
        }
        
        self.active_sessions[consultation_id] = consultation
        return consultation
    
    def find_suitable_practitioners(self, patient_data: dict, symptom_analysis: dict) -> list:
        """Find suitable AYUSH practitioners based on symptoms and patient profile"""
        practitioners = [
            {
                "id": "DR001",
                "name": "Dr. Rajesh Kumar",
                "system": "Ayurveda",
                "specialization": "General Medicine",
                "experience": "15 years",
                "rating": 4.8,
                "available_slots": ["10:00 AM", "2:00 PM", "4:00 PM"],
                "consultation_fee": 500,
                "languages": ["Hindi", "English", "Sanskrit"]
            },
            {
                "id": "DR002", 
                "name": "Dr. Priya Sharma",
                "system": "Siddha",
                "specialization": "Women's Health",
                "experience": "12 years",
                "rating": 4.9,
                "available_slots": ["11:00 AM", "3:00 PM", "5:00 PM"],
                "consultation_fee": 600,
                "languages": ["Tamil", "English", "Hindi"]
            },
            {
                "id": "DR003",
                "name": "Dr. Ahmed Hassan",
                "system": "Unani",
                "specialization": "Respiratory Disorders",
                "experience": "18 years", 
                "rating": 4.7,
                "available_slots": ["9:00 AM", "1:00 PM", "6:00 PM"],
                "consultation_fee": 550,
                "languages": ["Urdu", "Hindi", "English", "Arabic"]
            }
        ]
        
        # Filter based on symptom analysis and patient preference
        if symptom_analysis.get("found"):
            preferred_system = patient_data.get("preferred_system", "").lower()
            if preferred_system:
                practitioners = [p for p in practitioners if p["system"].lower() == preferred_system]
        
        return practitioners[:3]  # Return top 3 matches
    
    def generate_prescription_template(self, symptom_analysis: dict) -> dict:
        """Generate prescription template based on AI analysis"""
        if not symptom_analysis.get("found"):
            return {}
        
        template = {
            "diagnosis": f"{symptom_analysis['symptom']} - Multiple system analysis",
            "systems_consulted": [],
            "recommendations": {
                "medicines": [],
                "lifestyle": [],
                "diet": [],
                "follow_up": "1 week"
            },
            "precautions": [
                "Consult qualified AYUSH practitioner before starting treatment",
                "Monitor symptoms and report any adverse effects",
                "Continue existing medications unless advised otherwise"
            ]
        }
        
        for system in symptom_analysis.get("systems", []):
            template["systems_consulted"].append(system["system"])
            template["recommendations"]["medicines"].extend(system["medicines"])
            template["recommendations"]["lifestyle"].extend(system["lifestyle_recommendations"])
        
        return template

if __name__ == "__main__":
    print(" Telemedicine Integration Ready!")
    print(" ABHA integration active")
    print("‍ Virtual consultation platform online")
