import os
print("Current working directory:", os.getcwd())
print("Files in current directory:")
for f in os.listdir('.'):
    print(f"  {f}")