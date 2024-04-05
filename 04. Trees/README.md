# Trees

## Glossary

- [Binary Tree Node](#binary-tree-node)
- [Operations]()
  - [Searching]()
    - [Recursive Search]()
    - [Iterative Search]()
    - [Successor and Predecessor]()
  - [Insertion]()
  - [Deletion]()
- [Traversal]()
- [Priority Queue Operations]()

## Binary Tree Node

```Python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
```

## Traversal

#### pre-order: root->left->right

<details>
<summary>recursive</summary>

```Python
def preorder_traversal(self, root: 'TreeNode'):
    if not root:
        return

    self.process(root)
    self.preorder_traversal(root.left)
    self.preorder_traversal(root.right)
```

</details>

<details>
<summary>iterative</summary>

```Python
def preorder_traversal(self, root: 'TreeNode'):
    if not root:
        return []
    stack = [root]
    while stack:
        root = stack.pop()
        self.process(root)

        if root.right: # push right child first because of FILO
            stack.append(root.right)
        if root.left:
            stack.append(root.left)
```

</details>

### In-order: left->root->right

<details>
<summary>recursive</summary>

```Python
def inorder_traversal(self, root: 'TreeNode'):
    if not root:
        return

    self.inorder_traversal(root.left)
    self.process(root)
    self.inorder_traversal(root.right)
```

</details>

<details>
<summary>iterative</summary>

```Python
def inorder_traversal(self, root: 'TreeNode'):
    stack = []
    while stack or root:
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop()
        self.process(root)
        root = root.right
```

</details>

### Post-order: left->right->root

<details>
<summary>recursive</summary>

```Python
def postorder_traversal(self, root: 'TreeNode'):
    if not root:
        return

    self.postorder_traversal(root.left)
    self.postorder_traversal(root.right)
    self.process(root)
```

</details>

<details>

<summary>iterative</summary>

```Python
def postorder_traversal(self, root: 'TreeNode'):
    if not root:
        return []
    stack = [root]
    # used to record whether left or right child has been visited
    last = None

    while stack:
        root = stack[-1]
        if not root.left and not root.right or last and (root.left == last or root.right == last):
            self.process(root)
            stack.pop()
            last = root
        else:
            if root.right: # push right first because of FILO
                stack.append(root.right)
            if root.left:
                stack.append(root.left)
```

</details>

### Level-order

<details>
<summary>recursive</summary>

```Python
def level_order_traversal(self, root: 'TreeNode') -> list[list[int]]:
    res = []

    def dfs(root, level):
        if not root:
            return
        if len(res) < level + 1:
            res.append([])
        '''
        add current node logic here
        '''
        self.process_logic(root)

        res[level].append(root.val)
        dfs(root.left, level + 1)
        dfs(root.right, level + 1)

    dfs(root, 0)
    return res
```

</details>

<details>
<summary>iterative</summary>

```Python
def level_order_traversal_iteratively(self, root: 'TreeNode'):
    if not root:
        return []
    queue = deque([root])
    while queue:
        for _ in range(len(queue)):
            cur = queue.popleft()
            self.process(cur)

            if cur.left:
                queue.append(cur.left)
            if cur.right:
                queue.append(cur.right)

            # Level logic goes here
```

</details>

### Tree Iterators

- [Reference](https://www.alexbowe.com/iterative-tree-traversal/)

#### Rationale

- The iterative approach using a stack is generally more **memory-efficient** than
  recursion, as it **avoids the overhead of function calls**.
- This can be beneficial for very large trees where _recursive calls_ might lead to stack **overflow errors**.

```Python
def tree_iterator(root, expand):
  stack = [(False, root)]
  while stack:
    expanded, curr = stack.pop()
    if not curr: continue
    if expanded: yield curr
    else: stack.extend([(x is curr, x) for x in reversed(expand(curr))])

preorder  = lambda root: tree_iterator(root, lambda node: (node,      node.left,  node.right))
inorder   = lambda root: tree_iterator(root, lambda node: (node.left, node,       node.right))
postorder = lambda root: tree_iterator(root, lambda node: (node.left, node.right, node))

# And their reversals
reverse_preorder  = lambda root: tree_iterator(root, lambda node: (node,       node.right, node.left))
reverse_inorder   = lambda root: tree_iterator(root, lambda node: (node.right, node,       node.left))
reverse_postorder = lambda root: tree_iterator(root, lambda node: (node.right, node.left,  node))

# TODO: understand this code.
def level_order(root):
  q = [root]
  f = lambda x: [q.extend([x.left, x.right]),x][1]
  return (f(x) for x in q if x) # and not prune(x)
```

**Examples:**

#### Count the number of nodes in a Binary Tree

```Python
sum(1 for _ in inorder(root))
```

#### Count the number of leaf nodes in a Binary Tree

```Python
is_leaf = lambda x: not x.left and not x.right
sum(1 for x in inorder(root) if is_leaf(x)))
```

#### Verify that a Binary Tree is a Binary Search Tree

```Python
from itertools import pairwise
is_ordered = lambda xs: all(a<=b for a,b in pairwise(xs))
is_ordered(x.value for x in inorder(tree))
```

#### Flatten a Binary Search Tree into an Ordered List

```Python
[x.value for x in inorder(tree)]
```

#### Find the kth-smallest element of Binary Search Tree

```Python
next(x.value for i,x in enumerate(inorder(root)) if i == k-1)
```

#### Find the kth-largest element of Binary Search Tree

```Python
next(x.value for i,x in enumerate(reverse_inorder(root)) if i == k-1)
```

#### Check if a Binary Tree is Symmetrical

```Python
from itertools import zip_longest
all(a.value == b.value for a,b in zip_longest(inorder(tree_a), reverse_inorder(tree_b)))
```

#### TODO: Height of Binary Tree
