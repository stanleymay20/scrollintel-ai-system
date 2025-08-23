import os
#!/usr/bin/env python3
"""
ScrollIntel Demo Test Script
Test the running ScrollIntel system with sample data
"""

import requests
import json
import time
from pathlib import Path

# ScrollIntel API base URL
BASE_URL = os.getenv("API_URL", os.getenv("API_URL", "http://localhost:8000"))

def test_health_check():
    """Test if ScrollIntel is running"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            health_data = response.json()
            print("✅ ScrollIntel Health Check:")
            print(f"   Status: {health_data['status']}")
            print(f"   Version: {health_data['version']}")
            print(f"   Features: {', '.join(health_data['features'])}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to ScrollIntel: {e}")
        return False

def test_agent_chat():
    """Test AI agent chat functionality"""
    try:
        # Test CTO agent
        chat_data = {
            "message": "What's the best approach for scaling a data pipeline to handle 1TB of daily data?",
            "agent_type": "cto"
        }
        
        response = requests.post(f"{BASE_URL}/api/agents/chat", json=chat_data)
        if response.status_code == 200:
            result = response.json()
            print("✅ CTO Agent Response:")
            print(f"   {result.get('response', 'No response')[:200]}...")
            return True
        else:
            print(f"❌ Agent chat failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Agent chat error: {e}")
        return False

def test_file_upload():
    """Test file upload functionality"""
    try:
        # Create sample CSV data
        sample_data = """name,age,salary,department
John Doe,30,75000,Engineering
Jane Smith,28,68000,Marketing
Bob Johnson,35,82000,Engineering
Alice Brown,32,71000,Sales
Charlie Wilson,29,69000,Marketing"""
        
        # Save to temporary file
        temp_file = Path("sample_data.csv")
        temp_file.write_text(sample_data)
        
        # Upload file
        with open(temp_file, 'rb') as f:
            files = {'file': ('sample_data.csv', f, 'text/csv')}
            response = requests.post(f"{BASE_URL}/api/files/upload", files=files)
        
        # Clean up
        temp_file.unlink()
        
        if response.status_code == 200:
            result = response.json()
            print("✅ File Upload Success:")
            print(f"   File ID: {result.get('file_id', 'Unknown')}")
            print(f"   Rows: {result.get('rows', 'Unknown')}")
            return result.get('file_id')
        else:
            print(f"❌ File upload failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ File upload error: {e}")
        return None

def test_data_analysis(file_id):
    """Test data analysis functionality"""
    if not file_id:
        return False
        
    try:
        analysis_data = {
            "file_id": file_id,
            "analysis_type": "summary"
        }
        
        response = requests.post(f"{BASE_URL}/api/analysis/analyze", json=analysis_data)
        if response.status_code == 200:
            result = response.json()
            print("✅ Data Analysis Success:")
            print(f"   Analysis: {str(result)[:200]}...")
            return True
        else:
            print(f"❌ Data analysis failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Data analysis error: {e}")
        return False

def main():
    """Run ScrollIntel demo tests"""
    print("🚀 ScrollIntel Demo Test Suite")
    print("=" * 50)
    
    # Test 1: Health Check
    if not test_health_check():
        print("\n❌ ScrollIntel is not running. Please start it first:")
        print("   python run_simple.py")
        return
    
    print("\n" + "=" * 50)
    
    # Test 2: Agent Chat
    print("\n🤖 Testing AI Agent Chat...")
    test_agent_chat()
    
    print("\n" + "=" * 50)
    
    # Test 3: File Upload
    print("\n📁 Testing File Upload...")
    file_id = test_file_upload()
    
    print("\n" + "=" * 50)
    
    # Test 4: Data Analysis
    print("\n📊 Testing Data Analysis...")
    test_data_analysis(file_id)
    
    print("\n" + "=" * 50)
    print("\n🎉 Demo Complete!")
    print("\n📋 Next Steps:")
    print("   1. Visit http://localhost:8000 to see the web interface")
    print("   2. Upload your own datasets")
    print("   3. Chat with different AI agents")
    print("   4. Explore the dashboard and visualizations")
    print("\n🚀 ScrollIntel is ready for production use!")

if __name__ == "__main__":
    main()