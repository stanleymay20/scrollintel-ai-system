#!/usr/bin/env python3
"""
Requirements Validation Script
Validates that all imported packages are included in requirements files.
"""

import ast
import os
import sys
from pathlib import Path
from typing import Set, List, Dict
import re

def extract_imports_from_file(file_path: Path) -> Set[str]:
    """Extract all import statements from a Python file."""
    imports = set()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the AST
        try:
            tree = ast.parse(content)
        except SyntaxError:
            print(f"Warning: Could not parse {file_path}")
            return imports
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return imports

def get_all_python_files(directory: Path) -> List[Path]:
    """Get all Python files in the directory recursively."""
    python_files = []
    
    for root, dirs, files in os.walk(directory):
        # Skip certain directories
        skip_dirs = {'.git', '__pycache__', '.pytest_cache', 'node_modules', '.venv', 'venv', '.env'}
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    
    return python_files

def parse_requirements_file(req_file: Path) -> Set[str]:
    """Parse requirements file and extract package names."""
    packages = set()
    
    if not req_file.exists():
        return packages
    
    try:
        with open(req_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Extract package name (before >= or == or [)
                    match = re.match(r'^([a-zA-Z0-9_-]+)', line)
                    if match:
                        packages.add(match.group(1).lower().replace('-', '_'))
    except Exception as e:
        print(f"Error reading {req_file}: {e}")
    
    return packages

def get_standard_library_modules() -> Set[str]:
    """Get a set of standard library module names."""
    # This is a subset of common standard library modules
    return {
        'os', 'sys', 'json', 'time', 'datetime', 'pathlib', 'typing', 'asyncio',
        'logging', 'collections', 'itertools', 'functools', 'operator', 'math',
        'random', 'string', 'io', 'tempfile', 'shutil', 'subprocess', 'threading',
        'multiprocessing', 'queue', 'socket', 'urllib', 'http', 'email', 'base64',
        'hashlib', 'hmac', 'secrets', 'uuid', 'decimal', 'fractions', 'statistics',
        'enum', 'dataclasses', 'contextlib', 'warnings', 'traceback', 'inspect',
        'gc', 'weakref', 'copy', 'pickle', 'shelve', 'sqlite3', 'csv', 'configparser',
        'argparse', 'getopt', 'readline', 'rlcompleter', 'cmd', 'shlex', 'glob',
        'fnmatch', 'linecache', 'fileinput', 'filecmp', 'stat', 'platform',
        'ctypes', 'struct', 'codecs', 'unicodedata', 'stringprep', 'textwrap',
        'locale', 'calendar', 'zoneinfo', 'heapq', 'bisect', 'array', 'types',
        'abc', 'numbers', 'cmath', 're', 'difflib', 'pprint', 'reprlib',
        'unittest', 'doctest', 'test', 'bdb', 'faulthandler', 'pdb', 'profile',
        'pstats', 'timeit', 'trace', 'tracemalloc', 'dis', 'pickletools',
        'formatter', 'gettext', 'locale', 'calendar', 'mailcap', 'mailbox',
        'mimetypes', 'quopri', 'uu', 'html', 'xml', 'webbrowser', 'cgi',
        'cgitb', 'wsgiref', 'ftplib', 'poplib', 'imaplib', 'nntplib', 'smtplib',
        'smtpd', 'telnetlib', 'socketserver', 'xmlrpc', 'ipaddress', 'audioop',
        'aifc', 'sunau', 'wave', 'chunk', 'colorsys', 'imghdr', 'sndhdr',
        'ossaudiodev', 'getpass', 'curses', 'platform', 'errno', 'ctypes',
        'mmap', 'winreg', 'winsound', 'posix', 'pwd', 'spwd', 'grp', 'crypt',
        'termios', 'tty', 'pty', 'fcntl', 'pipes', 'resource', 'nis', 'syslog',
        'optparse', 'imp', 'importlib', 'keyword', 'pkgutil', 'modulefinder',
        'runpy', 'parser', 'ast', 'symtable', 'symbol', 'token', 'tokenize',
        'tabnanny', 'pyclbr', 'py_compile', 'compileall', 'dis', 'pickletools',
        'distutils', 'ensurepip', 'venv', 'zipapp'
    }

def main():
    """Main validation function."""
    print("🔍 Validating Requirements Files...")
    
    # Get project root
    project_root = Path(__file__).parent
    
    # Parse requirements files
    main_req = parse_requirements_file(project_root / "requirements.txt")
    security_req = parse_requirements_file(project_root / "security" / "requirements.txt")
    ai_req = parse_requirements_file(project_root / "ai_data_readiness_requirements.txt")
    api_req = parse_requirements_file(project_root / "ai_data_readiness" / "api" / "requirements.txt")
    
    all_requirements = main_req | security_req | ai_req | api_req
    
    print(f"📦 Found {len(all_requirements)} packages in requirements files")
    
    # Get all Python files
    python_files = get_all_python_files(project_root)
    print(f"🐍 Scanning {len(python_files)} Python files...")
    
    # Extract all imports
    all_imports = set()
    for py_file in python_files:
        imports = extract_imports_from_file(py_file)
        all_imports.update(imports)
    
    # Filter out standard library modules and local modules
    stdlib_modules = get_standard_library_modules()
    local_modules = {'scrollintel', 'ai_data_readiness', 'security', 'tests', 'scripts'}
    
    third_party_imports = set()
    for imp in all_imports:
        if imp not in stdlib_modules and imp not in local_modules:
            third_party_imports.add(imp.lower().replace('-', '_'))
    
    print(f"📚 Found {len(third_party_imports)} third-party imports")
    
    # Find missing packages
    missing_packages = third_party_imports - all_requirements
    
    # Package name mappings (import name -> package name)
    package_mappings = {
        'cv2': 'opencv_python',
        'sklearn': 'scikit_learn',
        'jose': 'python_jose',
        'PIL': 'pillow',
        'yaml': 'pyyaml',
        'dotenv': 'python_dotenv',
        'multipart': 'python_multipart',
        'socketio': 'python_socketio',
        'jwt': 'pyjwt',
        'bcrypt': 'bcrypt',
        'cryptography': 'cryptography',
        'prometheus_client': 'prometheus_client',
        'structlog': 'structlog',
        'elasticsearch': 'elasticsearch',
        'kafka': 'kafka_python',
        'docker': 'docker',
        'kubernetes': 'kubernetes',
        'boto3': 'boto3',
        'azure': 'azure',
        'google': 'google_cloud',
        'huggingface_hub': 'huggingface_hub',
        'wandb': 'wandb',
        'mlflow': 'mlflow',
        'neptune': 'neptune_client',
        'comet_ml': 'comet_ml',
        'tensorboard': 'tensorboard',
        'gradio': 'gradio',
        'streamlit': 'streamlit',
        'dash': 'dash',
        'flask': 'flask',
        'django': 'django',
        'celery': 'celery',
        'airflow': 'apache_airflow',
        'prefect': 'prefect',
        'dask': 'dask',
        'ray': 'ray',
        'spark': 'pyspark',
    }
    
    # Check for mapped packages
    actually_missing = set()
    for pkg in missing_packages:
        mapped_name = package_mappings.get(pkg, pkg)
        if mapped_name not in all_requirements:
            actually_missing.add(pkg)
    
    if actually_missing:
        print(f"\n❌ Missing packages in requirements files:")
        for pkg in sorted(actually_missing):
            mapped = package_mappings.get(pkg, pkg)
            print(f"   - {pkg} (install as: {mapped})")
        
        print(f"\n📝 Add these to requirements.txt:")
        for pkg in sorted(actually_missing):
            mapped = package_mappings.get(pkg, pkg)
            print(f"{mapped}>=1.0.0")
    else:
        print("\n✅ All imported packages are included in requirements files!")
    
    # Check for unused packages (packages in requirements but not imported)
    unused_packages = all_requirements - third_party_imports
    
    # Filter out known packages that might not be directly imported
    known_indirect = {
        'setuptools', 'wheel', 'pip', 'uvicorn', 'gunicorn', 'pytest', 'black',
        'flake8', 'mypy', 'pre_commit', 'factory_boy', 'pytest_asyncio',
        'pytest_cov', 'alembic', 'psycopg2_binary', 'asyncpg', 'aiosqlite',
        'python_multipart', 'email_validator', 'pydantic_settings',
        'strawberry_graphql', 'python_jose', 'passlib', 'python_dotenv',
        'croniter', 'schedule', 'backoff', 'safetensors', 'accelerate',
        'xformers', 'diffusers', 'dice_ml'
    }
    
    unused_packages = unused_packages - known_indirect
    
    if unused_packages:
        print(f"\n⚠️  Potentially unused packages in requirements:")
        for pkg in sorted(unused_packages):
            print(f"   - {pkg}")
    
    print(f"\n📊 Summary:")
    print(f"   - Total Python files scanned: {len(python_files)}")
    print(f"   - Third-party imports found: {len(third_party_imports)}")
    print(f"   - Packages in requirements: {len(all_requirements)}")
    print(f"   - Missing packages: {len(actually_missing)}")
    print(f"   - Potentially unused packages: {len(unused_packages)}")

if __name__ == "__main__":
    main()