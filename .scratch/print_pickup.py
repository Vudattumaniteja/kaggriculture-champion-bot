import inspect
import kaggle_environments.envs.kaggriculture.kaggriculture as k

src = inspect.getsource(k)
lines = src.splitlines()

for i, line in enumerate(lines):
    if 'if op == "DROP":' in line:
        for j in range(i, min(len(lines), i + 70)):
            print(lines[j])
        break
