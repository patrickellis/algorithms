# Strings

## Table of Contents

- [String Hashing](#string-hashing)
  - [Rabin-karp](#rabin-karp-algorithm)
  - [Knuth-Morris-Pratt](#knuth-morris-pratt-algorithm)
- [Trie/Prefix Tree](#trie-prefix-tree)
- [Suffix Tree](#suffix-tree)

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

### Rabin-karp Algorithm

- [Reference](https://cp-algorithms.com/string/rabin-karp.html)

_Problem: given two strings - a pattern $s$ and a text $t$, determine if the pattern appears in the text and if it does, enumerate all its occurrences._

**Algorithm**:

1. Calculate the hash of the pattern $s$.
2. Calculate hash values for all the prefixes of the text $t$ (Sliding window).
3. Compare each substring of length $|s|$ with the pattern. _This will take a total of $O(|t|)$ time._

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

## Trie/Prefix Tree

- [Wikipedia](https://en.wikipedia.org/wiki/Trie)

## Suffix Tree

- [Reference](https://favtutor.com/blogs/ukkonen-algorithm-suffix-tree)
- [Wikipedia](https://en.wikipedia.org/wiki/Suffix_tree)
