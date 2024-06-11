# Strings

## Table of Contents

- [String Hashing](#string-hashing)
  - [Rabin-karp](#rabin-karp-algorithm)
  - [Knuth-Morris-Pratt](#knuth-morris-pratt-algorithm)
- [Trie/Prefix Tree](#trie-prefix-tree)
- [Suffix Tree](#suffix-tree)

> [!IMPORTANT]
> A string-matching overview can found [here](https://www-igm.univ-mlv.fr/~lecroq/string/node2.html).

## String Hashing

- [Reference](https://cp-algorithms.com/string/string-hashing.html)

The good and widely used way to define the hash of a string $s$ of length $n$ is:

$$
hash(s)=s[0]+s[1]\cdot p + s[2]\cdot{p^2} + ...+s[n-1]\cdot p^{n-1} \mod m \\
= \sum_{i=0}^{n-1}s[i]\cdot p^i \mod m,
$$

where $p$ and $m$ are some chosen, positive numbers. It is called a **polynomial rolling hash function**.

<details>
<summary>Choosing values for p and m</summary>

- $p$ is typically a prime number roughly equal to the number of characters in the input alphabet.  
   Thus, if the input is composed f only lowercase english letters, $p=31$ is a good choice.  
   If both uppercase and lowercase, $p=53$ is a possible choice.
- Obviously, $m$ should be a large number since the probability of two random strings colliding is $\approx\frac{1}{m}$.  
  **Practically, a good choice for $m$ is some large prime number.**
- We will be using $m=10^9+9$.
  This is large, yet small enough that we can multiply two values using 64-bit integers.

</details>

**Example**: Calculate the hash of a string $s$, which contains only lowercase english letters.

First convert each character of s to an integer. Here we use $a→1,b\rightarrow 2,...,z\rightarrow 26$.  
 _N.B. converting $a\rightarrow 0$ is not a good idea, because then the hashes of the strings $a,aa,aaa,...$ all evaluate to 0._

```Python
def compute_hash(s: str) -> int:
    p = 31
    m = 10^9 + 9
    hash_value = 0
    p_pow = 1
    for c in s:
        hash_value = (hash_value + (c-'a'+1) * p_pow) % m
        p_pow = (p_pow * p) % m
    return hash_value
```

### Brute Force

- [Resourec with  features, description, and brief code.](https://www-igm.univ-mlv.fr/~lecroq/string/node3.html#SECTION0030)

The brute force algorithm consists in checking, at all positions in the text between 0 and n-m, whether an occurrence of the pattern starts there or not. Then, after each attempt, it shifts the pattern by exactly one position to the right.

The brute force algorithm requires no preprocessing phase, and a constant extra space in addition to the pattern and the text. During the searching phase the text character comparisons can be done in any order. The time complexity of this searching phase is O(mn) (when searching for am-1b in an for instance). The expected number of text character comparisons is 2n.


```Python
def bf_search(pattern, text):
    """
    Brute force string matching algorithm.
    
    :param pattern: The pattern string to search for.
    :param text: The text string in which to search for the pattern.
    :return: A list of starting indices where the pattern is found in the text.
    """
    m = len(pattern)
    n = len(text)
    matches = []

    # Searching
    for j in range(n - m + 1):
        i = 0
        while i < m and pattern[i] == text[i + j]:
            i += 1
        if i >= m:
            matches.append(j)
    
    return matches

# Example usage
text = "hello world, welcome to the world of programming"
pattern = "world"
result = bf_search(pattern, text)
print(result)  # Output: [6, 23]
```

### Rabin-karp Algorithm

- [Reference](https://cp-algorithms.com/string/rabin-karp.html)
- [Algorithm Details, e.g.  features, description, and alternative code.](https://www-igm.univ-mlv.fr/~lecroq/string/node5.html#SECTION0050)

_Problem: given two strings - a pattern $s$ and a text $t$, determine if the pattern appears in the text and if it does, enumerate all its occurrences._

**Algorithm**:

1. Calculate the hash of the pattern $s$.
2. Calculate hash values for all the prefixes of the text $t$ (Sliding window).
3. Compare each substring of length $|s|$ with the pattern. _This will take a total of $O(|t|)$ time._



- Where m is a pattern, and n is us:
    - uses an hashing function;
    - preprocessing phase in O(m) time complexity and constant space;
    - searching phase in O(mn) time complexity;
    - O(n+m) expected running time.


**Time Complexity:** $O(|t|+|s|)$

- $O(|s|)$ to calculate the pattern hash and $O(|t|)$ to compare each substring of length $|s|$ with the pattern.

```Python
def rabin_karp(s: str, t: str):
    p = 31
    m = 1e9+9
    S, T = len(s), len(t)

    p_pow = [None]*max(S,T)
    p_pow[0] = 1
    for i in range(1,len(p_pow)):
        p_pow[i] = (p_pow[i-1]*p)%m

    h = [0]*(T+1)
    for i in range(T):
        h[i+1] = (h[i] + (t[i]-'a'+1)*p_pow[i]) % m

    h_s = 0
    for i in range(S):
        h_s = (h_s + (s[i]-'a'+1) * p_pow[i]) % m

    occurences = []
    for i in range(T-S+1):
        cur_h = (h[i+S]+m-h[i]) % m
        # TODO: understand order of operations below. Modulo vs. multiply.
        if cur_h == h_s * p_pow[i] % m:
            occurrences.append(i)

    return occurrences
```

### Knuth-Morris-Pratt Algorithm

- [Algorithm featurse, description, and code](https://www-igm.univ-mlv.fr/~lecroq/string/node8.html#SECTION0080).
- [Reference](https://cp-algorithms.com/string/prefix-function.html)

**Prefix function mathematical definition:**
$$\pi[i]=max_{k=0..i}\{k: s[0...k-1]=s[i-(k-1)...i]\}$$

<details>
<summary>Written definition and example</summary>

You are given a string $s$  of length $n$ . The prefix function for this string is defined as an array  
$\pi$  of length  $n$ , where $\pi[i]$  is the length of the longest proper prefix of the substring  
$s[0 \dots i]$  which is also a suffix of this substring. A proper prefix of a string is a prefix that is not equal to the string itself. By definition,  
$\pi[0] = 0$ .

For example, prefix function of string "abcabcd" is  
$[0, 0, 0, 1, 2, 3, 0]$ , and prefix function of string "aabaaab" is  
$[0, 1, 0, 1, 2, 2, 3]$ .

</details>

```Python
"""
An implementation of the Knuth-Morris-Pratt algorithm.
https://www-igm.univ-mlv.fr/~lecroq/string/node8.html#SECTION0080
"""

def kmp(text, pattern):
    """
    Given a pattern and a text, KMP finds all the places that the pattern
    is found in the text (even overlapping pattern matches).

    :param text: The text in which to search for the pattern.
    :param pattern: The pattern to search for in the text.
    :return: A list of starting indices where the pattern is found in the text.
    """
    matches = []
    if text is None or pattern is None:
        return matches

    m, n = len(pattern), len(text)
    if m > n:
        return matches

    lps = kmp_helper(pattern)

    i = j = 0  # index for text and pattern
    while i < n:
        if pattern[j] == text[i]:
            j += 1
            i += 1

        if j == m:
            matches.append(i - j)
            j = lps[j - 1]
        elif i < n and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

    return matches

def kmp_helper(pattern):
    """
    Preprocess the pattern to create the longest proper prefix array.

    :param pattern: The pattern to preprocess.
    :return: The longest proper prefix array.
    """
    m = len(pattern)
    lps = [0] * m
    length = 0  # length of the previous longest prefix suffix
    i = 1

    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1

    return lps

if __name__ == "__main__":
    matches = kmp("abababa", "aba")
    print(matches)  # Output: [0, 2, 4]

    matches = kmp("abc", "abcdef")
    print(matches)  # Output: []

    matches = kmp("P@TTerNabcdefP@TTerNP@TTerNabcdefabcdefabcdefabcdefP@TTerN", "P@TTerN")
    print(matches)  # Output: [0, 13, 20, 51]

```

#### Trivial Prefix Function Algorithm

Time complexity: $O(n^3)$

```Python
def prefix_function(s: str):
    n = len(s)
    pi = [None]*n
    for i in range(n):
        for k in range(i+1):
            if s[0:k] == s[i-k+1:i+1]:
                pi[i] = k
    return pi
```

#### Efficient Algorithm

- Read about the three optimizations [here](https://cp-algorithms.com/string/prefix-function.html#efficient-algorithm).
- ..and the applications of KMP [here](https://cp-algorithms.com/string/prefix-function.html#applications):
  - [Substring search](https://cp-algorithms.com/string/prefix-function.html#search-for-a-substring-in-a-string-the-knuth-morris-pratt-algorithm)
  - [Count occurrences of each prefix](https://cp-algorithms.com/string/prefix-function.html#counting-the-number-of-occurrences-of-each-prefix)
  - [Count distinct substrings in a string](https://cp-algorithms.com/string/prefix-function.html#the-number-of-different-substring-in-a-string)
  - [Compressing a string](https://cp-algorithms.com/string/prefix-function.html#compressing-a-string)

```Python
def prefix_function(s: str):
    n = len(s)
    pi = [None]*n
    for i in range(1,n):
        j = pi[i-1]
        while j > 0 and s[i]!=s[j]:
            j = pi[j-1]
        if s[i] == s[j]:
            j+=1
        pi[i] = j
    return pi
```


## Z-Algorithm


```Python
def calculate_z(text: str) -> list:
    """
    Calculates the Z-array of a given string.

    :param text: The string on which Z-array is computed.
    :return: A list which is the Z-array of the text.
    """
    if text is None:
        return []

    size = len(text)
    Z = [0] * size
    L, R, k = 0, 0, 0

    for i in range(size):
        if i == 0:
            Z[i] = size
        elif i > R:
            L = R = i
            while R < size and text[R - L] == text[R]:
                R += 1
            Z[i] = R - L
            R -= 1
        else:
            k = i - L
            if Z[k] < R - i + 1:
                Z[i] = Z[k]
            else:
                L = i
                while R < size and text[R - L] == text[R]:
                    R += 1
                Z[i] = R - L
                R -= 1

    return Z

Example usage: 

result = calculate_z("example")
print(result)
```

## SubstringVerificationSuffixArray

```Python
"""
This module demonstrates how to use a suffix array to determine if a pattern exists within a text.
This implementation has the advantage that once the suffix array is built, queries can be very fast.

Time complexity: O(n log n) for suffix array construction and O(m log n) time for individual
queries (where m is the query string length). As noted below, depending on the length of the string
(if it is very large), it may be faster to use the Knuth-Morris-Pratt (KMP) algorithm or, if you're 
doing a lot of queries on small strings, then Rabin-Karp in combination with a bloom filter.
"""
class SuffixArray:
    def __init__(self, text: str):
        self.ALPHABET_SZ = 256
        self.text = text
        self.T = [ord(c) for c in text]
        self.N = len(self.T)
        self.sa = [0] * self.N
        self.sa2 = [0] * self.N
        self.rank = [0] * self.N
        self.c = [0] * max(self.ALPHABET_SZ, self.N)
        self.construct()

    def construct(self):
        """
        Constructs the suffix array for the text.
        """
        T, N, ALPHABET_SZ = self.T, self.N, self.ALPHABET_SZ
        sa, sa2, rank, c = self.sa, self.sa2, self.rank, self.c

        for i in range(N):
            c[rank[i] = T[i]] += 1
        for i in range(1, ALPHABET_SZ):
            c[i] += c[i - 1]
        for i in range(N - 1, -1, -1):
            sa[--c[T[i]]] = i

        p = 1
        while p < N:
            r = 0
            for i in range(N - p, N):
                sa2[r] = i
                r += 1
            for i in range(N):
                if sa[i] >= p:
                    sa2[r] = sa[i] - p
                    r += 1

            c = [0] * ALPHABET_SZ
            for i in range(N):
                c[rank[i]] += 1
            for i in range(1, ALPHABET_SZ):
                c[i] += c[i - 1]
            for i in range(N - 1, -1, -1):
                sa[--c[rank[sa2[i]]]] = sa2[i]

            sa2[sa[0]] = r = 0
            for i in range(1, N):
                if not (rank[sa[i - 1]] == rank[sa[i]] and
                        sa[i - 1] + p < N and
                        sa[i] + p < N and
                        rank[sa[i - 1] + p] == rank[sa[i] + p]):
                    r += 1
                sa2[sa[i]] = r

            rank, sa2 = sa2, rank
            if r == N - 1:
                break
            ALPHABET_SZ = r + 1
            p <<= 1

    def contains(self, substr: str) -> bool:
        """
        Checks if the substring exists in the text using the suffix array.

        :param substr: The substring to search for.
        :return: True if the substring exists, otherwise False.
        """
        if substr is None:
            return False
        if substr == "":
            return True

        T, sa, N = self.T, self.sa, self.N
        lo, hi = 0, N - 1
        substr_len = len(substr)

        while lo <= hi:
            mid = (lo + hi) // 2
            suffix_index = sa[mid]
            suffix_str = ''.join(chr(T[i]) for i in range(suffix_index, min(suffix_index + substr_len, N)))
            cmp = (suffix_str > substr) - (suffix_str < substr)

            if cmp == 0:
                return True
            elif cmp < 0:
                lo = mid + 1
            else:
                hi = mid - 1

        return False

# Example usage
if __name__ == "__main__":
    pattern = "hello world"
    text = "hello lemon Lennon wallet world tree cabbage hello world teapot calculator"
    sa = SuffixArray(text)
    print(sa.contains(pattern))  # Output: True
    print(sa.contains("this pattern does not exist"))  # Output: False
```

## Trie/Prefix Tree

- [Wikipedia](https://en.wikipedia.org/wiki/Trie)

```Python
# note: using a class is only necessary if you want to store data at each node.
# otherwise, you can implement a trie using only hash maps.
class TrieNode:
    def __init__(self):
        # you can store data at nodes if you wish
        self.data = None
        self.children = {}

def fn(words):
    root = TrieNode()
    for word in words:
        curr = root
        for c in word:
            if c not in curr.children:
                curr.children[c] = TrieNode()
            curr = curr.children[c]
        # at this point, you have a full word at curr
        # you can perform more logic here to give curr an attribute if you want
    
    return root
```

## Suffix Tree

- [Reference](https://favtutor.com/blogs/ukkonen-algorithm-suffix-tree)
- [Wikipedia](https://en.wikipedia.org/wiki/Suffix_tree)

##  Manachers Algorithms


```Python
"""
An implementation of Manacher's algorithm which can be used to find/count palindromic strings in
linear time. In particular, it finds the length of the maximal palindrome centered at each index.
"""

def manachers(s):
    """
    Manacher's algorithm finds the length of the longest palindrome centered at a specific index.
    Since even length palindromes have a center in between two characters we expand the string to insert
    those centers. For example, "abba" becomes "^#a#b#b#a#$" where the '#' sign represents the center of an even length string and
    '^' & '$' are the front and the back of the string respectively. The output of this function gives
    the diameter of each palindrome centered at each character in this expanded string.
    
    Example:
    manachers("abba") -> [0, 0, 1, 0, 1, 4, 1, 0, 1, 0, 0]
    """
    arr = pre_process(s)
    n = len(arr)
    p = [0] * n
    c = r = 0
    for i in range(1, n - 1):
        inv_i = 2 * c - i
        p[i] = min(r - i, p[inv_i]) if r > i else 0
        while arr[i + 1 + p[i]] == arr[i - 1 - p[i]]:
            p[i] += 1
        if i + p[i] > r:
            c = i
            r = i + p[i]
    return p

def pre_process(s):
    """
    Pre-process the string by injecting separator characters.
    We do this to account for even length palindromes, so we can assign them a unique center.
    
    Example:
    "abba" -> "^#a#b#b#a#$"
    """
    arr = ['^']
    for char in s:
        arr.append('#')
        arr.append(char)
    arr.append('#')
    arr.append('$')
    return arr

def find_palindrome_substrings(s):
    """
    Finds all the palindrome substrings found inside a string. It uses Manacher's algorithm to find the diameter
    of each palindrome centered at each position.
    
    :param s: The input string.
    :return: A set of all palindromic substrings.
    """
    S = list(s)
    centers = manachers(S)
    palindromes = set()

    for i in range(len(centers)):
        diameter = centers[i]
        if diameter >= 1:
            if i % 2 == 1:
                while diameter > 1:
                    index = (i - 1) // 2 - diameter // 2
                    palindromes.add(''.join(S[index:index + diameter]))
                    diameter -= 2
            else:
                while diameter >= 1:
                    index = (i - 2) // 2 - (diameter - 1) // 2
                    palindromes.add(''.join(S[index:index + diameter]))
                    diameter -= 2

    return palindromes

if __name__ == "__main__":
    s = "abbaabba"
    # Outputs: {'a', 'aa', 'abba', 'abbaabba', 'b', 'baab', 'bb', 'bbaabb'}
    print(find_palindrome_substrings(s))
```
