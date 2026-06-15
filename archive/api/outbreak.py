"""
Outbreak Prediction Engine Endpoint
REST API for epidemic forecasting and regional surveillance
"""
import json
from datetime import datetime, timedelta
from typing import Dict, List

def handler(request):
    """Outbreak prediction engine endpoint"""
    
    method = request.get('method', 'GET')
    path = request.get('path', '/')
    
    if path == '/api/outbreak/predict' and method == 'POST':
        # Main outbreak prediction endpoint
        try:
            body = json.loads(request.get('body', '{}'))
            region = body.get('region', 'all')
            time_window = body.get('time_window', 30)  # days
            
            # Simulated outbreak prediction engine
            predictions = predict_outbreaks(region, time_window)
            
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps(predictions)
            }
        except Exception as e:
            return {
                'statusCode': 500,
                'body': json.dumps({"error": str(e)})
            }
    
    elif path == '/api/outbreak/risk-assessment' and method == 'POST':
        # Regional risk assessment
        try:
            body = json.loads(request.get('body', '{}'))
            region = body.get('region', 'india')
            
            risk = assess_regional_risk(region)
            
            return {
                'statusCode': 200,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps(risk)
            }
        except Exception as e:
            return {
                'statusCode': 500,
                'body': json.dumps({"error": str(e)})
            }
    
    elif path == '/api/outbreak/timeline' and method == 'GET':
        # Get outbreak timeline
        region = request.get('query', {}).get('region', ['india'])[0]
        
        timeline = get_outbreak_timeline(region)
        
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps(timeline)
        }
    
    elif path == '/api/outbreak/alerts' and method == 'GET':
        # Get active outbreak alerts
        region = request.get('query', {}).get('region', ['all'])[0]
        severity = request.get('query', {}).get('severity', ['all'])[0]
        
        alerts = get_outbreak_alerts(region, severity)
        
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps(alerts)
        }
    
    elif path == '/api/outbreak/heatmap' and method == 'GET':
        # Get regional heatmap
        heatmap = get_regional_heatmap()
        
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps(heatmap)
        }
    
    elif path == '/api/outbreak/forecast' and method == 'POST':
        # Advanced forecast with ML
        try:
            body = json.loads(request.get('body', '{}'))
            disease = body.get('disease')
            region = body.get('region', 'india')
            days = body.get('forecast_days', 30)
            
            forecast = generate_forecast(disease, region, days)
            
            return {
                'statusCode': 200,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps(forecast)
            }
        except Exception as e:
            return {
                'statusCode': 500,
                'body': json.dumps({"error": str(e)})
            }
    
    else:
        return {
            'statusCode': 404,
            'body': json.dumps({"error": "Endpoint not found"})
        }


def predict_outbreaks(region: str, time_window: int) -> Dict:
    """Predict potential outbreaks in region"""
    
    # Simulated outbreak predictions based on clustering
    predictions = {
        "region": region,
        "time_window_days": time_window,
        "prediction_date": datetime.utcnow().isoformat(),
        "outbreaks_detected": 3,
        "high_risk_areas": [],
        "moderate_risk_areas": [],
        "low_risk_areas": []
    }
    
    # Simulated high-risk outbreak
    if region in ['maharashtra', 'all']:
        predictions["high_risk_areas"].append({
            "location": "Mumbai Metropolitan Area",
            "disease": "COVID-19",
            "risk_score": 82,
            "risk_level": "critical",
            "cases_predicted": 3500,
            "confidence": 0.87,
            "factors": [
                "Population density concentration",
                "Recent travel increase",
                "Variant spread patterns",
                "Low vaccination in sub-populations"
            ],
            "recommendation": "Activate surveillance protocol, prepare ICU capacity"
        })
    
    # Moderate risk
    if region in ['karnataka', 'all']:
        predictions["moderate_risk_areas"].append({
            "location": "Bangalore Urban",
            "disease": "Dengue",
            "risk_score": 58,
            "risk_level": "high",
            "cases_predicted": 850,
            "confidence": 0.72,
            "factors": [
                "Monsoon water accumulation",
                "Mosquito breeding sites increase",
                "Seasonal pattern"
            ],
            "recommendation": "Increase mosquito control measures"
        })
    
    # Low risk
    predictions["low_risk_areas"].append({
        "location": "Tamil Nadu Coast",
        "disease": "Influenza",
        "risk_score": 32,
        "risk_level": "low",
        "cases_predicted": 200,
        "confidence": 0.65,
        "factors": ["Seasonal variation", "Wind patterns"],
        "recommendation": "Routine surveillance"
    })
    
    # Summary statistics
    predictions["summary"] = {
        "total_risk_score": 172,
        "average_risk": 57.3,
        "highest_risk_disease": "COVID-19",
        "highest_risk_location": "Mumbai Metropolitan Area",
        "recommendation_priority": "critical"
    }
    
    return predictions


def assess_regional_risk(region: str) -> Dict:
    """Assess regional epidemic risk"""
    
    risk_matrix = {
        "maharashtra": {
            "current_risk": 75,
            "trend": "increasing",
            "major_diseases": ["COVID-19", "Dengue", "Malaria"],
            "populations_vulnerable": 3500000,
            "health_system_capacity": 0.65
        },
        "karnataka": {
            "current_risk": 58,
            "trend": "stable",
            "major_diseases": ["Dengue", "Malaria"],
            "populations_vulnerable": 2100000,
            "health_system_capacity": 0.72
        },
        "tamil_nadu": {
            "current_risk": 42,
            "trend": "decreasing",
            "major_diseases": ["Dengue"],
            "populations_vulnerable": 1800000,
            "health_system_capacity": 0.80
        },
        "delhi": {
            "current_risk": 65,
            "trend": "increasing",
            "major_diseases": ["COVID-19", "Influenza"],
            "populations_vulnerable": 2800000,
            "health_system_capacity": 0.58
        }
    }
    
    if region == 'all':
        all_risks = {}
        for reg, data in risk_matrix.items():
            all_risks[reg] = data
        all_risks['overall_risk'] = 60.0
        return all_risks
    
    return risk_matrix.get(region, {
        "current_risk": 0,
        "trend": "unknown",
        "major_diseases": [],
        "populations_vulnerable": 0,
        "health_system_capacity": 0.5
    })


def get_outbreak_timeline(region: str) -> Dict:
    """Get historical outbreak timeline"""
    
    timeline = {
        "region": region,
        "timeline_start": (datetime.utcnow() - timedelta(days=90)).isoformat(),
        "timeline_end": datetime.utcnow().isoformat(),
        "events": [
            {
                "date": (datetime.utcnow() - timedelta(days=60)).isoformat(),
                "disease": "COVID-19",
                "event": "Outbreak detected",
                "cases_reported": 250,
                "severity": "high"
            },
            {
                "date": (datetime.utcnow() - timedelta(days=45)).isoformat(),
                "disease": "COVID-19",
                "event": "Peak cases recorded",
                "cases_reported": 1250,
                "severity": "critical"
            },
            {
                "date": (datetime.utcnow() - timedelta(days=20)).isoformat(),
                "disease": "Dengue",
                "event": "Seasonal spike detected",
                "cases_reported": 450,
                "severity": "moderate"
            },
            {
                "date": (datetime.utcnow() - timedelta(days=5)).isoformat(),
                "disease": "COVID-19",
                "event": "Cases stabilizing",
                "cases_reported": 85,
                "severity": "low"
            }
        ]
    }
    
    return timeline


def get_outbreak_alerts(region: str, severity: str) -> Dict:
    """Get active outbreak alerts"""
    
    all_alerts = [
        {
            "alert_id": "ALERT-001",
            "region": "maharashtra",
            "disease": "COVID-19",
            "severity": "critical",
            "active": True,
            "issued": datetime.utcnow().isoformat(),
            "message": "Critical outbreak in Mumbai Metropolitan Area",
            "action_required": True
        },
        {
            "alert_id": "ALERT-002",
            "region": "karnataka",
            "disease": "Dengue",
            "severity": "high",
            "active": True,
            "issued": (datetime.utcnow() - timedelta(days=2)).isoformat(),
            "message": "High dengue activity in Bangalore region",
            "action_required": True
        },
        {
            "alert_id": "ALERT-003",
            "region": "tamil_nadu",
            "disease": "Dengue",
            "severity": "moderate",
            "active": True,
            "issued": (datetime.utcnow() - timedelta(days=5)).isoformat(),
            "message": "Seasonal dengue increase expected",
            "action_required": False
        }
    ]
    
    # Filter by region
    if region != 'all':
        all_alerts = [a for a in all_alerts if a['region'] == region]
    
    # Filter by severity
    if severity != 'all':
        all_alerts = [a for a in all_alerts if a['severity'] == severity]
    
    return {
        "total_alerts": len(all_alerts),
        "active_alerts": len([a for a in all_alerts if a['active']]),
        "alerts": all_alerts
    }


def get_regional_heatmap() -> Dict:
    """Get regional epidemic heatmap"""
    
    heatmap = {
        "timestamp": datetime.utcnow().isoformat(),
        "regions": {
            "maharashtra": {"risk_score": 75, "color": "red", "cases": 3500},
            "delhi": {"risk_score": 65, "color": "orange-red", "cases": 2800},
            "karnataka": {"risk_score": 58, "color": "orange", "cases": 2100},
            "west_bengal": {"risk_score": 48, "color": "yellow", "cases": 1500},
            "tamil_nadu": {"risk_score": 42, "color": "yellow-green", "cases": 1200},
            "others": {"risk_score": 30, "color": "green", "cases": 500}
        },
        "color_scheme": {
            "red": "Critical (70-100)",
            "orange-red": "High (60-70)",
            "orange": "Moderate (40-60)",
            "yellow": "Elevated (20-40)",
            "yellow-green": "Low (10-20)",
            "green": "Minimal (0-10)"
        }
    }
    
    return heatmap


def generate_forecast(disease: str, region: str, days: int) -> Dict:
    """Generate disease forecast using ML model"""
    
    # Simulated ML forecast
    base_cases = {
        "COVID-19": 500,
        "Dengue": 300,
        "Influenza": 150,
        "Malaria": 100
    }
    
    growth_rates = {
        "COVID-19": 1.15,
        "Dengue": 1.25,
        "Influenza": 0.95,
        "Malaria": 0.85
    }
    
    base = base_cases.get(disease, 100)
    growth = growth_rates.get(disease, 1.0)
    
    projections = []
    for day in range(1, days + 1):
        projected = int(base * (growth ** (day / 7)))  # Weekly growth
        projections.append({
            "day": day,
            "date": (datetime.utcnow() + timedelta(days=day)).isoformat(),
            "projected_cases": projected,
            "confidence_interval": {
                "lower": int(projected * 0.8),
                "upper": int(projected * 1.2)
            }
        })
    
    return {
        "disease": disease,
        "region": region,
        "forecast_model": "Exponential Growth with Confidence Intervals",
        "forecast_horizon": days,
        "generated_at": datetime.utcnow().isoformat(),
        "accuracy_estimate": 0.78,
        "projections": projections,
        "summary": {
            "current_cases": base,
            "projected_cases_day_30": projections[-1]['projected_cases'] if projections else 0,
            "trend": "increasing" if growth > 1.0 else "decreasing"
        }
    }
