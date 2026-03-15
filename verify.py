import os
print("Verify Start")
print(f"CWD: {os.getcwd()}")
try:
    os.makedirs("verify_dir")
    print("Created dir")
except:
    print("Dir exists or fail")
print("Verify End")
