#!/usr/bin/env python3
"""
Setup script for LegalMind Enhanced Contract Analyzer
"""

import os
import sys
import subprocess
import asyncio
from pathlib import Path

def run_command(command, description):
    """Run a shell command with error handling"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def create_directory_structure():
    """Create necessary directory structure"""
    directories = [
        "contracts",
        "azure_services", 
        "uploads",
        "logs",
        "exports",
        "cache",
        "tests",
        "docs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")

def install_python_dependencies():
    """Install Python dependencies"""
    commands = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install -r requirements.txt", "Installing Python dependencies"),
        ("python -m spacy download en_core_web_sm", "Downloading spaCy English model"),
        ("python -c \"import nltk; nltk.download('punkt'); nltk.download('vader_lexicon')\"", "Downloading NLTK data")
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    return True

def setup_database():
    """Setup database schema"""
    commands = [
        ("alembic init alembic", "Initializing Alembic"),
        ("alembic revision --autogenerate -m 'Initial migration'", "Creating initial migration"),
        ("alembic upgrade head", "Running database migrations")
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            print(f"⚠️  Database setup step failed: {description}")
            print("   You may need to configure your database connection first")

def create_env_file():
    """Create .env file from template if it doesn't exist"""
    if not os.path.exists('.env'):
        if os.path.exists('.env.example'):
            subprocess.run('cp .env.example .env', shell=True)
            print("📝 Created .env file from template")
            print("⚠️  Please update .env file with your actual configuration values")
        else:
            print("❌ .env.example file not found")

def setup_azure_services():
    """Setup Azure services"""
    print("\n🔷 Azure Services Setup")
    print("To enable Azure Text Analytics features:")
    print("1. Create an Azure account (free tier available)")
    print("2. Create a Text Analytics resource in Azure Portal")
    print("3. Get your endpoint and key from the Azure Portal")
    print("4. Update AZURE_TEXT_ANALYTICS_ENDPOINT and AZURE_TEXT_ANALYTICS_KEY in .env")
    print("5. Run: az login (if you have Azure CLI installed)")

def run_tests():
    """Run basic tests to verify setup"""
    print("\n🧪 Running basic tests...")
    
    test_commands = [
        ("python -c \"from contracts.patterns import ContractPatterns; print('✅ Patterns module loaded')\"", "Testing patterns module"),
        ("python -c \"from contracts.explainability import ExplainabilityEngine; print('✅ Explainability engine loaded')\"", "Testing explainability module"),
        ("python -c \"from azure_services.explainability_service import AzureExplainabilityService; print('✅ Azure service loaded')\"", "Testing Azure service module"),
    ]
    
    for command, description in test_commands:
        run_command(command, description)

def main():
    """Main setup function"""
    print("🚀 Setting up LegalMind Enhanced Contract Analyzer")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create directory structure
    print("\n📁 Creating directory structure...")
    create_directory_structure()
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not install_python_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Create environment file
    print("\n⚙️  Setting up configuration...")
    create_env_file()
    
    # Setup database (optional)
    print("\n🗄️  Setting up database...")
    setup_database()
    
    # Azure setup information
    setup_azure_services()
    
    # Run tests
    run_tests()
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Update your .env file with actual API keys and configuration")
    print("2. Configure your database connection")
    print("3. Set up Azure Text Analytics (optional but recommended)")
    print("4. Run the application: uvicorn main:app --reload")
    print("5. Navigate to http://localhost:8000/docs for API documentation")
    
    print("\n📚 Documentation:")
    print("- API docs: http://localhost:8000/docs")
    print("- ReDoc: http://localhost:8000/redoc")
    print("- Health check: http://localhost:8000/health")

if __name__ == "__main__":
    main()
