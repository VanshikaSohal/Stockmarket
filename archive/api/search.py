"""
Terminology Search API for Vercel Deployment
"""
import json
from urllib.parse import parse_qs

def handler(request):
    """Search medical terminology endpoint"""
    
    # Parse query parameters
    query_string = request.get('query', '')
    if hasattr(request, 'args'):
        query = request.args.get('q', '')
        limit = int(request.args.get('limit', 10))
    else:
        # Parse from query string manually for Vercel
        params = parse_qs(query_string)
        query = params.get('q', [''])[0]
        limit = int(params.get('limit', [10])[0])
    
    # Mock search results - replace with actual search logic
    search_results = {
        "query": query,
        "results": [
            {
                "code": "T40.1X1A",
                "system": "ICD-11",
                "display": f"Search result for: {query}",
                "accuracy": 96.7,
                "category": "Biomedicine"
            },
            {
                "code": "NAM001",
                "system": "NAMASTE",
                "display": f"NAMASTE code for: {query}",
                "accuracy": 94.3,
                "category": "Ayurveda"
            },
            {
                "code": "WHO-TM2-001",
                "system": "WHO-ICD11-TM2",
                "display": f"WHO Traditional Medicine code for: {query}",
                "accuracy": 92.1,
                "category": "Traditional Medicine"
            }
        ][:limit],
        "total": 3,
        "limit": limit,
        "timestamp": "2025-09-11T00:00:00Z"
    }
    
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
        },
        'body': json.dumps(search_results)
    }
