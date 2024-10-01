class SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.t = [0] * (4 * self.n)
        self.build(data, 1, 0, self.n - 1)

    def build(self, data, v, tl, tr):
        if tl == tr:
            self.t[v] = data[tl]
        else:
            tm = (tl + tr) // 2
            self.build(data, 2 * v, tl, tm)
            self.build(data, 2 * v + 1, tm + 1, tr)
            self.t[v] = self.t[2 * v] + self.t[2 * v + 1]

    def sum(self, v, tl, tr, l, r):
        if l > r:
            return 0
        if l == tl and r == tr:
            return self.t[v]
        tm = (tl + tr) // 2
        return self.sum(2 * v, tl, tm, l, min(r, tm)) + self.sum(
            2 * v + 1, tm + 1, tr, max(l, tm + 1), r
        )

    def update(self, v, tl, tr, pos, new_val):
        if tl == tr:
            self.t[v] = new_val
        else:
            tm = (tl + tr) // 2
            if pos <= tm:
                self.update(2 * v, tl, tm, pos, new_val)
            else:
                self.update(
                    2 * v + 1, tm + 1, tr, pos, new_val
                )
            self.t[v] = self.t[2 * v] + self.t[2 * v + 1]

    def query_sum(self, l, r):
        return self.sum(1, 0, self.n - 1, l, r)

    def update_value(self, pos, new_val):
        self.update(1, 0, self.n - 1, pos, new_val)


# Example usage
data = [1, 3, 5, 7, 9, 11]
segment_tree = SegmentTree(data)

# Query the sum of the range [1, 4)
print(segment_tree.query_sum(1, 4))  # Output: 24 (3 + 5 + 7 + 9)

# Update the value at index 2 to 6
segment_tree.update_value(2, 6)

# Query the sum of the range [1, 4) again after the update
print(segment_tree.query_sum(1, 4))  # Output: 25 (3 + 6 + 7 + 9)
