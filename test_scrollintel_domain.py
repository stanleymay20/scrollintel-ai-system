#!/usr/bin/env python3
"""Test ScrollIntel domain deployment"""

import requests

def test_scrollintel_domain():
    """Test if ScrollIntel is deployed correctly"""
    
    urls = [
        "https://scrollintel.com",
        "https://www.scrollintel.com", 
        "https://api.scrollintel.com/health",
        "https://api.scrollintel.com/docs"
    ]
    
    print("Testing ScrollIntel domain deployment...")
    
    for url in urls:
        try:
            print(f"Testing {url}...")
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print(f"SUCCESS: {url} - {response.status_code}")
            else:
                print(f"WARNING: {url} - {response.status_code}")
        except Exception as e:
            print(f"ERROR: {url} - {e}")
    
    print("Domain test complete!")

if __name__ == "__main__":
    test_scrollintel_domain()
