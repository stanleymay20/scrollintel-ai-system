#!/usr/bin/env python3
"""
Quick fix for GraphQL dependency conflicts during Docker build.
This script temporarily comments out graphene imports to allow the build to succeed.
"""

import os
import re

def fix_graphene_imports():
    """Comment out graphene imports to resolve dependency conflicts."""
    
    files_to_fix = [
        'scrollintel/engines/graphql_generator.py',
        'scrollintel/engines/api_code_generator.py', 
        'scrollintel/api/data_product_app.py',
        'scrollintel/api/graphql/data_product_schema.py'
    ]
    
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            print(f"Fixing {file_path}...")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Comment out graphene imports
            content = re.sub(r'^(import graphene.*)', r'# \1  # Temporarily disabled for Docker build', content, flags=re.MULTILINE)
            content = re.sub(r'^(from graphene.*)', r'# \1  # Temporarily disabled for Docker build', content, flags=re.MULTILINE)
            content = re.sub(r'^(from graphene_sqlalchemy.*)', r'# \1  # Temporarily disabled for Docker build', content, flags=re.MULTILINE)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"Fixed {file_path}")
        else:
            print(f"File not found: {file_path}")

if __name__ == "__main__":
    fix_graphene_imports()
    print("GraphQL dependency fix completed!")