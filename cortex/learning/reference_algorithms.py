"""Reference algorithms and data structures (Fibonera-authored, project-local).

Every function carries deterministic doctests; the expansion pipeline verifies
each one by actual sandbox execution before it can become a training pair.
"""
from __future__ import annotations


def binary_search(items, target):
    """Return the index of target in a sorted list, or -1.

    >>> binary_search([1, 3, 5, 7, 9], 7)
    3
    >>> binary_search([1, 3, 5], 4)
    -1
    >>> binary_search([], 1)
    -1
    """
    lo, hi = 0, len(items) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if items[mid] == target:
            return mid
        if items[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def merge_sorted(a, b):
    """Merge two sorted lists into one sorted list.

    >>> merge_sorted([1, 3, 5], [2, 4])
    [1, 2, 3, 4, 5]
    >>> merge_sorted([], [1])
    [1]
    """
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


def quicksort(items):
    """Return a new sorted list (quicksort, non-mutating).

    >>> quicksort([3, 1, 2])
    [1, 2, 3]
    >>> quicksort([])
    []
    >>> quicksort([2, 2, 1])
    [1, 2, 2]
    """
    if len(items) <= 1:
        return list(items)
    pivot = items[len(items) // 2]
    less = [x for x in items if x < pivot]
    equal = [x for x in items if x == pivot]
    greater = [x for x in items if x > pivot]
    return quicksort(less) + equal + quicksort(greater)


def fib_fast(n):
    """Fast Fibonacci via fast doubling.

    >>> fib_fast(0)
    0
    >>> fib_fast(1)
    1
    >>> fib_fast(10)
    55
    """
    def _fib_pair(k):
        if k == 0:
            return (0, 1)
        a, b = _fib_pair(k >> 1)
        c = a * (2 * b - a)
        d = a * a + b * b
        if k & 1:
            return (d, c + d)
        return (c, d)
    return _fib_pair(n)[0]


def gcd(a, b):
    """Greatest common divisor.

    >>> gcd(12, 18)
    6
    >>> gcd(7, 13)
    1
    """
    while b:
        a, b = b, a % b
    return abs(a)


def sieve_primes(limit):
    """All primes <= limit via the sieve of Eratosthenes.

    >>> sieve_primes(10)
    [2, 3, 5, 7]
    >>> sieve_primes(2)
    [2]
    """
    if limit < 2:
        return []
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    p = 2
    while p * p <= limit:
        if is_prime[p]:
            for m in range(p * p, limit + 1, p):
                is_prime[m] = False
        p += 1
    return [i for i, ok in enumerate(is_prime) if ok]


def dijkstra(graph, start):
    """Shortest distances from start in a dict graph {node: {nbr: weight}}.

    >>> dijkstra({'a': {'b': 1}, 'b': {'c': 2}, 'c': {}}, 'a') == {'a': 0, 'b': 1, 'c': 3}
    True
    """
    import heapq

    dist = {node: float("inf") for node in graph}
    dist[start] = 0
    heap = [(0, start)]
    while heap:
        d, node = heapq.heappop(heap)
        if d > dist[node]:
            continue
        for nbr, w in graph[node].items():
            nd = d + w
            if nd < dist[nbr]:
                dist[nbr] = nd
                heapq.heappush(heap, (nd, nbr))
    return dist


def bfs_levels(graph, start):
    """BFS level map from start over dict adjacency {node: [nbrs]}.

    >>> bfs_levels({'a': ['b', 'c'], 'b': [], 'c': ['d'], 'd': []}, 'a') == {'a': 0, 'b': 1, 'c': 1, 'd': 2}
    True
    """
    from collections import deque

    levels = {start: 0}
    q = deque([start])
    while q:
        node = q.popleft()
        for nbr in graph[node]:
            if nbr not in levels:
                levels[nbr] = levels[node] + 1
                q.append(nbr)
    return levels


def topological_order(edges, nodes):
    """Kahn topological sort; edges = (u, v) meaning u before v.

    >>> topological_order([('a', 'b'), ('b', 'c')], ['a', 'b', 'c'])
    ['a', 'b', 'c']
    >>> topological_order([], ['x'])
    ['x']
    """
    indeg = {n: 0 for n in nodes}
    adj = {n: [] for n in nodes}
    for u, v in edges:
        adj[u].append(v)
        indeg[v] += 1
    ready = sorted(n for n in nodes if indeg[n] == 0)
    order = []
    while ready:
        n = ready.pop(0)
        order.append(n)
        for m in adj[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                ready.append(m)
    if len(order) != len(nodes):
        raise ValueError("cycle detected")
    return order


class UnionFind:
    """Disjoint-set with union by size and path compression.

    >>> uf = UnionFind(4)
    >>> uf.union(0, 1); uf.union(2, 3)
    True
    >>> uf.union(1, 2); uf.find(0) == uf.find(3)
    True
    """

    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True


def longest_common_subsequence(a, b):
    """Length of the LCS of two strings.

    >>> longest_common_subsequence('abcde', 'ace')
    3
    >>> longest_common_subsequence('abc', 'def')
    0
    """
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def edit_distance(a, b):
    """Levenshtein edit distance.

    >>> edit_distance('kitten', 'sitting')
    3
    >>> edit_distance('', 'abc')
    3
    """
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def is_balanced(s):
    """True if brackets (), [], {} are balanced and properly nested.

    >>> is_balanced('([{}])')
    True
    >>> is_balanced('(]')
    False
    >>> is_balanced('')
    True
    """
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for ch in s:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack


def prefix_function(s):
    """KMP prefix function.

    >>> prefix_function('abacaba')
    [0, 0, 1, 0, 1, 2, 3]
    """
    pi = [0] * len(s)
    for i in range(1, len(s)):
        k = pi[i - 1]
        while k > 0 and s[i] != s[k]:
            k = pi[k - 1]
        if s[i] == s[k]:
            k += 1
        pi[i] = k
    return pi


def quickselect_kth(items, k):
    """k-th smallest (0-indexed) via quickselect.

    >>> quickselect_kth([3, 1, 2], 0)
    1
    >>> quickselect_kth([3, 1, 2], 2)
    3
    """
    if not 0 <= k < len(items):
        raise IndexError("k out of range")
    items = list(items)
    lo = 0
    hi = len(items) - 1
    while True:
        pivot = items[(lo + hi) // 2]
        i, j = lo, hi
        while i <= j:
            while items[i] < pivot:
                i += 1
            while items[j] > pivot:
                j -= 1
            if i <= j:
                items[i], items[j] = items[j], items[i]
                i += 1
                j -= 1
        if k <= j:
            hi = j
        elif k >= i:
            lo = i
        else:
            return items[k]


def matrix_multiply(a, b):
    """Multiply two matrices (lists of lists).

    >>> matrix_multiply([[1, 2]], [[3], [4]])
    [[11]]
    """
    if not a or not b or len(a[0]) != len(b):
        raise ValueError("dimension mismatch")
    cols = len(b[0])
    out = []
    for row in a:
        acc = [0] * cols
        for k, rk in enumerate(row):
            for j in range(cols):
                acc[j] += rk * b[k][j]
        out.append(acc)
    return out


def word_ladder_length(begin, end, words):
    """Minimum transformation sequence length to reach end from begin.

    >>> word_ladder_length('hit', 'cog', {'hot', 'dot', 'dog', 'lot', 'log', 'cog'})
    5
    >>> word_ladder_length('hit', 'cog', {'hot', 'dot'})
    0
    """
    if end not in words:
        return 0
    frontier = [(begin, 1)]
    seen = {begin}
    alpha = "abcdefghijklmnopqrstuvwxyz"
    while frontier:
        word, depth = frontier.pop(0)
        if word == end:
            return depth
        for i in range(len(word)):
            for c in alpha:
                nxt = word[:i] + c + word[i + 1:]
                if nxt in words and nxt not in seen:
                    seen.add(nxt)
                    frontier.append((nxt, depth + 1))
    return 0


class Trie:
    """Prefix tree with insert/contains/prefix checks.

    >>> t = Trie()
    >>> t.insert('cat')
    >>> t.contains('cat'), t.contains('ca'), t.starts_with('ca')
    (True, False, True)
    """

    def __init__(self):
        self.children = {}
        self.word = False

    def insert(self, word):
        node = self
        for ch in word:
            node = node.children.setdefault(ch, Trie())
        node.word = True

    def contains(self, word):
        node = self._walk(word)
        return node is not None and node.word

    def starts_with(self, prefix):
        return self._walk(prefix) is not None

    def _walk(self, s):
        node = self
        for ch in s:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node


def knapsack_max_value(weights, values, capacity):
    """0/1 knapsack maximum value.

    >>> knapsack_max_value([1, 3, 4], [15, 20, 30], 4)
    35
    >>> knapsack_max_value([], [], 5)
    0
    """
    best = [0] * (capacity + 1)
    for w, v in zip(weights, values):
        for c in range(capacity, w - 1, -1):
            best[c] = max(best[c], best[c - w] + v)
    return best[capacity]


def roman_to_int(s):
    """Convert a Roman numeral to an integer.

    >>> roman_to_int('IX')
    9
    >>> roman_to_int('MCMXCIV')
    1994
    """
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    for i, ch in enumerate(s):
        if i + 1 < len(s) and vals[ch] < vals[s[i + 1]]:
            total -= vals[ch]
        else:
            total += vals[ch]
    return total


def flatten_nested(nested):
    """Flatten arbitrarily nested lists.

    >>> flatten_nested([1, [2, [3, []], 4]])
    [1, 2, 3, 4]
    """
    out = []
    for item in nested:
        if isinstance(item, list):
            out.extend(flatten_nested(item))
        else:
            out.append(item)
    return out


def int_to_roman(num):
    """Integer to Roman numeral (1..3999).

    >>> int_to_roman(9)
    'IX'
    >>> int_to_roman(1994)
    'MCMXCIV'
    """
    table = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"),
             (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"),
             (5, "V"), (4, "IV"), (1, "I")]
    out = []
    for value, sym in table:
        while num >= value:
            out.append(sym)
            num -= value
    return "".join(out)
