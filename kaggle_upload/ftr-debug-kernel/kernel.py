import os
print("cwd:", os.getcwd())
print("listing /kaggle:")
for root, dirs, files in os.walk('/kaggle'):
    depth = root.count(os.sep)
    if depth > 3:
        dirs[:] = []
        continue
    print(root, dirs, files[:10])
