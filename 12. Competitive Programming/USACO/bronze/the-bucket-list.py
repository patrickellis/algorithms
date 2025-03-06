import sys
sys.stdin = open('blist.in', 'r')
sys.stdout = open('blist.out', 'w')
N = int(input())
C = [tuple(map(int, input().split())) for _ in range(N)]

events = []

for s, t, b in C:
    events.append((s,b,'start'))
    events.append((t,b,'end'))

result = 0
count = 0
events.sort(key=lambda x: (x[0], 0 if x[2] == 'end' else 1))
for time, buckets, e_type in events:
    if e_type == "start":
        count += buckets
    else:
        count -= buckets
    result = max(result, count)

print(str(result))

