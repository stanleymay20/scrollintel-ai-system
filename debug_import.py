"""Debug import issues."""

try:
    print("Importing failure_prevention...")
    from scrollintel.core.failure_prevention import failure_prevention
    print("✓ failure_prevention imported")
    
    print("Importing user_experience_protection...")
    from scrollintel.core.user_experience_protection import ux_protector
    print("✓ ux_protector imported")
    
    print("Importing failure_ux_integration module...")
    import scrollintel.core.failure_ux_integration as fui
    print("✓ Module imported")
    
    print("Available items:", [x for x in dir(fui) if not x.startswith('_')])
    
    print("Checking for FailureUXIntegrator class...")
    if hasattr(fui, 'FailureUXIntegrator'):
        print("✓ FailureUXIntegrator class found")
        
        print("Creating instance...")
        integrator = fui.FailureUXIntegrator()
        print("✓ Instance created")
        
    else:
        print("✗ FailureUXIntegrator class not found")
    
    print("Checking for failure_ux_integrator instance...")
    if hasattr(fui, 'failure_ux_integrator'):
        print("✓ failure_ux_integrator instance found")
    else:
        print("✗ failure_ux_integrator instance not found")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()