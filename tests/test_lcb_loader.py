from code_solver.data.lcb_loader import LCBLoader

loader = LCBLoader()
problems = loader.load()
print(problems[0])
