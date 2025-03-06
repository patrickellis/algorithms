import sys

sys.stdin = open('traffic.in','r')
sys.stdout = open('traffic.out','w')

N = int(input())
data = [input().split() for _ in range(N)]

def solve(data):
    N = len(data)
    cmin, cmax = 0, float('inf')
    fmin, fmax = -1, -1
    for i in range(N):
        ramp, l, r = data[i]
        l, r = int(l), int(r)
        if ramp == 'on':
            cmin += l
            cmax += r
        elif ramp == 'off':
            cmin -= r
            cmax -= l
        else:
            if fmin == fmax == -1:
                fmin, fmax = l, r - cmin
            if r < cmax:
                cmax = r
            if l > cmin:
                cmin = l
    return fmin, fmax, cmin, cmax

fmin, fmax, omin, omax = solve(data)
print(f"{fmin} {fmax}")
print(f"{omin} {omax}")



