import sys

sys.stdin = open('blocks.in','r')
sys.stdout = open('blocks.out','w')

N = int(input())
data = [input().split() for _ in range(N)]

def solve(data):

fmin, fmax, omin, omax = solve(data)

print(f"{fmin} {fmax}")
print(f"{omin} {omax}")



