import asyncio
from scrollintel.api.routes.stakeholder_mapping_routes import identify_stakeholders

async def test_api():
    context = {
        'organization_name': 'Test Corp',
        'board_members': [
            {
                'id': 'board_001',
                'name': 'John Smith',
                'title': 'Board Chair',
                'industry_experience': ['technology'],
                'expertise': ['strategy'],
                'education': ['MBA'],
                'previous_roles': ['CEO'],
                'achievements': ['IPO'],
                'contact_preferences': {}
            }
        ],
        'executives': [],
        'investors': [],
        'advisors': []
    }
    
    result = await identify_stakeholders(context)
    print(f"API test successful! Found {result['total_count']} stakeholders")
    if result['stakeholders']:
        s = result['stakeholders'][0]
        print(f"First stakeholder: {s['name']} - {s['title']}")

if __name__ == "__main__":
    asyncio.run(test_api())