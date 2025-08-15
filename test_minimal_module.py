print("Module starting...")

from enum import Enum

print("Defining ProfileLevel...")

class ProfileLevel(Enum):
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"

print("ProfileLevel defined")

class MetadataExtractor:
    def __init__(self):
        pass

print("MetadataExtractor defined")
print("Module complete")