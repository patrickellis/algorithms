with open("shuffle.in", "r") as f:
    N = int(f.readline())
    S = list(map(int, f.readline().split()))
    ID = list(map(int, f.readline().split()))
print(S)
print(ID)
print('='*80)
A = list(range(N))

for i in range(3):
    print(f"[{i} shuffles] {A}")
    temp = [-1] * N
    for i in range(N):
        temp[S[i]-1] = A[i]
    A = temp
print(f'[3 shuffles] {A}')

orig = [0]*N
for i in range(N):
    orig[A[i]] = i

result = [str(ID[orig[i]]) for i in range(N)]


with open("shuffle.out", "w") as f:
    f.write("\n".join(result))

