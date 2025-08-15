try:
    print("Importing module...")
    import ai_data_readiness.core.metadata_extractor
    print("Module imported successfully")
    print("Module file:", ai_data_readiness.core.metadata_extractor.__file__)
    print("Module contents:", [x for x in dir(ai_data_readiness.core.metadata_extractor) if not x.startswith('_')])
    
    # Try to access specific classes
    try:
        from ai_data_readiness.core.metadata_extractor import ProfileLevel
        print("ProfileLevel imported successfully")
    except Exception as e:
        print(f"Failed to import ProfileLevel: {e}")
        
    try:
        from ai_data_readiness.core.metadata_extractor import MetadataExtractor
        print("MetadataExtractor imported successfully")
    except Exception as e:
        print(f"Failed to import MetadataExtractor: {e}")
        
except Exception as e:
    print(f"Error importing module: {e}")
    import traceback
    traceback.print_exc()