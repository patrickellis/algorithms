class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_leaf = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_leaf = True

    def search(self, word: str) -> bool:
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_leaf

    def starts_with(self, prefix: str) -> bool:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True


trie = Trie()
trie.insert("apple")
print(trie.search("apple"))  # Returns True
print(trie.search("app"))  # Returns False
print(trie.starts_with("app"))  # Returns True
trie.insert("app")
print(trie.search("app"))  # Returns True
