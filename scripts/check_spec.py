import kaggle_environments

env = kaggle_environments.make("kaggriculture")
spec = env.specification
print("Action space keys/doc:")
print(spec["action"])
