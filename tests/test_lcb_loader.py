from code_solver.data.lcb_loader import LCBLoader

loader = LCBLoader()
problems = loader.load()

p = problems[11]

for tc in p.private_tests:
    print(tc)
