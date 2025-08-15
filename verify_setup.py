"""
Verification script for ScrollIntel Core setup
"""
import os
import sys

def check_file_exists(filepath, description):
    """Check if a file exists and report"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} (missing)")
        return False

def check_directory_structure():
    """Check the ScrollIntel Core directory structure"""
    print("🔍 Checking ScrollIntel Core directory structure...")
    
    base_dir = "scrollintel_core"
    files_to_check = [
        (f"{base_dir}/main.py", "Main FastAPI application"),
        (f"{base_dir}/config.py", "Configuration management"),
        (f"{base_dir}/database.py", "Database setup"),
        (f"{base_dir}/models.py", "Database models"),
        (f"{base_dir}/requirements.txt", "Python dependencies"),
        (f"{base_dir}/docker-compose.yml", "Docker Compose configuration"),
        (f"{base_dir}/Dockerfile", "Docker container definition"),
        (f"{base_dir}/.env.example", "Environment configuration template"),
        (f"{base_dir}/README.md", "Documentation"),
        (f"{base_dir}/agents/orchestrator.py", "Agent orchestrator"),
        (f"{base_dir}/agents/base.py", "Base agent class"),
        (f"{base_dir}/agents/cto_agent.py", "CTO Agent"),
        (f"{base_dir}/agents/data_scientist_agent.py", "Data Scientist Agent"),
        (f"{base_dir}/agents/ml_engineer_agent.py", "ML Engineer Agent"),
        (f"{base_dir}/agents/bi_agent.py", "BI Agent"),
        (f"{base_dir}/agents/ai_engineer_agent.py", "AI Engineer Agent"),
        (f"{base_dir}/agents/qa_agent.py", "QA Agent"),
        (f"{base_dir}/agents/forecast_agent.py", "Forecast Agent"),
        (f"{base_dir}/api/routes.py", "API routes"),
        (f"{base_dir}/init-scripts/01-init-database.sql", "Database initialization"),
        (f"{base_dir}/frontend/package.json", "Frontend package configuration"),
        (f"{base_dir}/start.sh", "Linux/Mac startup script"),
        (f"{base_dir}/start.bat", "Windows startup script"),
    ]
    
    all_good = True
    for filepath, description in files_to_check:
        if not check_file_exists(filepath, description):
            all_good = False
    
    return all_good

def check_agent_implementations():
    """Check if all 7 core agents are implemented"""
    print("\n🤖 Checking core agent implementations...")
    
    agents_dir = "scrollintel_core/agents"
    required_agents = [
        "cto_agent.py",
        "data_scientist_agent.py", 
        "ml_engineer_agent.py",
        "bi_agent.py",
        "ai_engineer_agent.py",
        "qa_agent.py",
        "forecast_agent.py"
    ]
    
    all_agents_present = True
    for agent_file in required_agents:
        agent_path = os.path.join(agents_dir, agent_file)
        agent_name = agent_file.replace("_agent.py", "").replace("_", " ").title()
        if not check_file_exists(agent_path, f"{agent_name} Agent"):
            all_agents_present = False
    
    return all_agents_present

def main():
    """Main verification function"""
    print("🚀 ScrollIntel Core Setup Verification")
    print("=" * 50)
    
    structure_ok = check_directory_structure()
    agents_ok = check_agent_implementations()
    
    print("\n📊 Verification Summary:")
    print("=" * 30)
    
    if structure_ok and agents_ok:
        print("🎉 All checks passed! ScrollIntel Core is properly set up.")
        print("\n📋 Next steps:")
        print("1. Copy scrollintel_core/.env.example to scrollintel_core/.env")
        print("2. Edit .env file with your API keys and configuration")
        print("3. Run: cd scrollintel_core && ./start.sh dev (Linux/Mac)")
        print("   Or: cd scrollintel_core && start.bat dev (Windows)")
        print("4. Access the API at: http://localhost:8001")
        print("5. View API docs at: http://localhost:8001/docs")
        return True
    else:
        print("❌ Some files are missing. Please check the setup.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)