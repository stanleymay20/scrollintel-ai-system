import traceback

try:
    print("Reading file...")
    with open('scrollintel/engines/crisis_communication_integration.py', 'r') as f:
        code = f.read()
    
    print(f"File length: {len(code)} characters")
    
    # Create a new namespace
    namespace = {'__name__': '__main__'}
    
    print("Executing code...")
    exec(code, namespace)
    
    print("Execution successful")
    print("Namespace keys:", [k for k in namespace.keys() if not k.startswith('__')])
    
    if 'CrisisCommunicationIntegration' in namespace:
        print("Class found!")
        cls = namespace['CrisisCommunicationIntegration']
        print("Class:", cls)
        
        # Try to instantiate
        instance = cls()
        print("Instance created successfully")
    else:
        print("Class NOT found in namespace")
        
except Exception as e:
    print(f"Error: {e}")
    print("Full traceback:")
    traceback.print_exc()