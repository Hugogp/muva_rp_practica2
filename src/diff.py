import numpy as np

files = [
    "../Competicion2.txt",
    # "../Competicion2_150.txt",
    "../Competicion2_200.txt",
]

contents = []

for file in files:
    values = np.loadtxt(file, dtype=str)

    contents.append(values)

for i in range(len(contents[0])):
    c1 = contents[0][i]
    c2 = contents[1][i]
    # c3 = contents[2][i]

    if c1 != c2:  # or c2 != c3 or c1 != c3:
        print(f"Diff: {i}: [{c1}, {c2}]")
