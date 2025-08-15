try:
    from ai_data_readiness.core.metadata_extractor import MetadataExtractor
    print("MetadataExtractor imported successfully")
except Exception as e:
    print(f"Error importing MetadataExtractor: {e}")
    import traceback
    traceback.print_exc()

try:
    from ai_data_readiness.core.metadata_extractor import ProfileLevel
    print("ProfileLevel imported successfully")
except Exception as e:
    print(f"Error importing ProfileLevel: {e}")

try:
    from ai_data_readiness.core.metadata_extractor import DatasetProfile
    print("DatasetProfile imported successfully")
except Exception as e:
    print(f"Error importing DatasetProfile: {e}")