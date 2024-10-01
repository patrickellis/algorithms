x = (10**2) + (2**6) + 13  # random int

for i in range(31, -1, -1):
    if x & (1 << i):
        print(1, end="")
    else:
        print(0, end="")
