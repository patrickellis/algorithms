with open("lostcow.in", "r") as f:
    x, y = map(int, f.readline().split())

is_left = x < y
cond = lambda x: x >= y if is_left else x <= y
step = 1
start = x
d = 0
prev = 0


while not cond(x):
    d += prev
    x = start + step
    d += abs(step)
    prev = abs(step)
    step *= -2
d -= abs(x - y)
with open("lostcow.out", "w") as f:
    f.write(str(d))

