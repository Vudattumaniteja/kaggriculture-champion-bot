import inspect
import kaggle_environments.envs.kaggriculture.kaggriculture as k

src = inspect.getsource(k)
for line in src.splitlines():
    if "op == " in line:
        print(line)
