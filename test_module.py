try:
    import ai_data_readiness.core.metadata_extractor
    print("Module imported successfully")
    print("Module contents:", dir(ai_data_readiness.core.metadata_extractor))
except Exception as e:
    print(f"Error importing module: {e}")
    import traceback
    traceback.print_exc()