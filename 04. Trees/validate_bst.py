from itertools import pairwise

is_ordered = lambda x: all(a <= b for a,b in pairwise(x))
is_ordered(x.val for x in inorder(root))
