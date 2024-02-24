# Algorithms

Glossary

- Array
  - Binary Search
  - Sorting Algorithms
  - Two Pointers
    - Algorithms
      - Tortoise and Hare
    - Patterns
      - Running from both ends of an array
        - Reverse String
        - Is Palindrome
        - Next Permutation
        - Two Sum (Sorted)
        - N-Sum
      - Fast and Slow Pointers
        - String Compression
        - Remove Duplicates from Sorted Input
        - Find the Duplicate Number
      - Running from the beginning of two arrays 
        - Merge sorted arrays
      - Sliding Window
        - Minimum Window Substring
        - Permutation in String
  - Intervals


## Array


### Binary Search

- [Reading](https://leetcode.com/discuss/general-discussion/786126/python-powerful-ultimate-binary-search-template-solved-many-problems)

**Template:**

```Python
def binary_search(array) -> int:
    def condition(value) -> bool:
        pass

    left, right = min(search_space), max(search_space) # could be [0, n], [1, n] etc. Depends on problem
    while left < right:
        mid = left + (right - left) // 2
        if condition(mid):
            right = mid
        else:
            left = mid + 1
    return left
```

### Two Pointers

> [!NOTE]
> Strings are arrays of characters, therefore there are some String algorithms.

#### Algorithms

##### Tortoise and Hare

> [!IMPORTANT]
> Used for **Cycle Detection** in Linked lists and Arrays.

- [Reading](https://cp-algorithms.com/others/tortoise_and_hare.html)

**Template:**

```Python
def hasCycle(listNode head):
    slow = head
    fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

def getCycle(listNode head) -> int:
"""Returns the starting point of a cycle in a Linked list."""
    if not hasCycle(head):
        return -1
    
    slow = head
    fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    else:
        return None

    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
        
    return slow
```

#### Running from both ends of the array

##### Reverse String

```Python
def reverseString(self, s: list[str]) -> None:
    l, r = 0, len(s)-1

    while l < r:
        s[l], s[r] = s[r], s[l]
        l, r = l+1, r-1
```

##### Is Palindrome

```Python
def isPalindrome(self, s: str) -> bool:
"""Returns True if string 's' is a Palindrome, else False."""
    s = s.lower()
    l, r = 0, len(s) - 1

    while l < r:
        while l < r and not s[l].isalnum():
            l += 1
        while r > l and not s[r].isalnum():
            r -= 1
        if s[l] != s[r]:
            return False
        l += 1
        r -= 1
    return True
```

##### Next Permutation

```Python
def nextPermutation(A: list[int]) -> None:
    """Computes the next permutation of an input array A.
    
    The permutation generated is the next lexicographically greatest permutation.

    The approach is:
    
        1) Find the first element A[i] that is not in ascending order, starting
           from the end of A.

        2) Find the first element A[j] whose value is less than A[i], starting
           from the end of A.

        3) Swap the two values.

        4) Reverse the order of all elements after index i.
    """

    i = len(A) - 2
    while i >= 0 and A[i] >= A[i+1]: i -= 1
    if i >= 0:
        j = len(A) - 1
        while A[j] <= A[i]: j -= 1
        A[j], A[i] = A[i], A[j]
    
    A[i+1:] = A[i+1:][::-1]

    return
```

##### Two Sum (Sorted)

```Python
def twoSum(self, A: list[int], target: int) -> list[int]:
"""Calculates two sum result for a sorted input."""
    l, r = 0, len(A) - 1

    while l < r: 
        s = A[l] + A[r]
        if s > target:
            r -= 1
        elif s < target:
            l += 1
        else:
            return [l + 1, r + 1]
    
    return []
```

##### N-Sum

```Python
def nsum(A: list[int], N: int, results: list[list[int]], result: list[int], target: int):
"""Generalized function that calculates solutions for n-Sum.

Usage:
    
    results = []
    target = 11
    nsum(A, 4, results, [], target)
"""
    if A[0] * N > target or A[-1] * N < target:
        return
    
    if N == 2:
        l, r = 0, len(A) - 1
        while l < r:
            two_sum = A[l] + A[r]
            if two_sum < target:
                l += 1
            elif two_sum > target:
                r -= 1
            else:
                results.append(result+[A[l], A[r]])
                l, r = l + 1, r - 1
                while l < r and A[l] == A[l-1]:
                    l += 1
                while r > l and A[r] == A[r+1]:
                    r -= 1
    else:
        for i in range(len(A)-N+1):
            if i == 0 or A[i] != A[i-1]:
                nsum(A[i+1:], N-1, results, result+[A[i]],target-A[i])

```

#### Fast and Slow Pointers

##### [String Compression](https://leetcode.com/problems/string-compression/)

- `l, r` (left, right) doesn't make sense for some problems. Use `i, j` instead.

```Python
def compress(chars: list[str]) -> int:
"""Given an array of characters, compress it.

The algorithm used is:

    Begin with an empty string s. 
    For each group of consecutive repeating characters in chars:

    If the group's length is 1, append the character to s.
    Otherwise, append the character followed by the group's length.
"""
    i, j, n = 0, 0, len(chars)
    
    while i < n:
        curr = chars[i]
        count = 0
        
        while i < n and chars[i] == curr:
            i += 1
            count += 1
        
        chars[j] = curr
        j += 1

        if count > 1:
            for digit in str(count):
                chars[j] = digit
                j += 1
    return j
```

##### [Remove Duplicates from sorted input](https://leetcode.com/problems/remove-duplicates-from-sorted-array/)

```Python
def removeDuplicates(nums: list[int]) -> int:
    j = 1
    for i in range(1, len(nums)):
        if nums[i] != nums[i-1]:
            nums[j] = nums[i]
            j += 1
    return j
```

##### [Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/)
- Algorithm: Tortoise and Hare

```Python
def findDuplicate(A: list[int]) -> int:
    def hasCycle():
    """Returns true when a cycle is found."""
        slow = A[0]
        fast = A[0]
        while True:
            slow = A[slow]
            fast = A[A[fast]]
            if slow == fast:
                return True
        
    if not hasCycle():
        return -1
    
    slow = A[0]
    while slow != fast:
        slow = A[slow]
        fast = A[fast]
        
    return slow
```

#### Running from the beginning of two arrays

##### [Merge sorted arrays](https://leetcode.com/problems/merge-sorted-array/)

```Python
def merge(self, A: list[int], m: int, B: list[int], n: int) -> None:
"""Merges two sorted arrays A and B.

A has additional space to hold the additional elements from B.

m = len(A)
n = len(B)
"""
    k = m+n-1
    i = m-1
    j = n-1
    while i >= 0 and j >= 0:
        if A[i] >= B[j]:
            A[k] = A[i]
            i -= 1
        else:
            A[k] = B[j]
            j -= 1
        k -= 1
    A[:j+1] = B[:j+1]
```

#### Sliding Window

> [!IMPORTANT]
> General template for substring search questions.

- [Sliding Window Substring Search Template](https://leetcode.com/problems/find-all-anagrams-in-a-string/discuss/92007/sliding-window-algorithm-template-to-solve-all-the-leetcode-substring-search-problem)

```Python
def slidingWindow(s: str, t: str) -> list[int]:
    m, n = len(s), len(t)
    c = collections.Counter(t)
    counter = len(c)
    l, r = 0, 0
    res = []
    while r < m:
        if s[r] in c:
            c[s[r]] -= 1
            if c[s[r]] == 0:
                counter -= 1
        if r-l+1 == n:
            if counter == 0:
                res.append(l)
            if s[l] in c:
                c[s[l]] += 1
                if c[s[l]] == 1:
                    counter += 1
            l += 1
        r += 1
    return res
```


###### [Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/)

```Python
def minWindow(s: str, t: str) -> str:
    m, n = len(s), len(t)
    if m < n:
        return ""
    
    l, r = 0, 0
    d = collections.Counter(t)
    c = len(d)
    head, length = 0, 0
    while r < m:
        if s[r] in d:
            d[s[r]] -= 1
            c -= d[s[r]] == 0
        while c == 0:
            if not length or r-l+1 < length:
                head = l
                length = r-l+1
            if s[l] in d:
                d[s[l]] += 1
                c += d[s[l]] == 1
            l += 1
        r += 1
    return s[head:head+length]
```

##### [Permutation in String](https://leetcode.com/problems/permutation-in-string/)

```Python
def checkInclusion(self, s1: str, s2: str) -> bool:
"""Returns True if a string s2 contains a permutation of s1, otherwise False."""
    d = dict.fromkeys(s1, 0)
    for c in s1: 
        d[c] += 1
    counter = len(d)

    l, r = 0, 0
    while r < len(s2):
        if s2[r] in d:
            d[s2[r]] -= 1
            counter -= d[s2[r]] == 0
        
        if r-l+1 == len(s1):
            if counter == 0:
                return True
            if s2[l] in d:
                d[s2[l]] += 1
                counter += d[s2[l]] == 1
            l += 1
        r += 1
    return False
```
