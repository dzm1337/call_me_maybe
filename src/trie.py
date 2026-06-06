class TrieNode:
    """A single node in the Trie,
    holding its children and end-of-word flag."""

    def __init__(self) -> None:
        self.children: dict[str, TrieNode] = {}
        self.is_end_of_word: bool = False


class Trie:
    """Prefix tree used to efficiently constrain
    decoding to valid function names."""

    def __init__(self) -> None:
        self.root: TrieNode = TrieNode()

    def insert(self, word: str) -> None:
        """Insert a word character by character into the trie."""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def get_allowed_next_chars(self, prefix: str) -> set[str]:
        """Return the set of valid next
        characters that extend the given prefix."""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return set()
            node = node.children[char]
        return set(node.children.keys())

    def is_prefix(self, text: str) -> bool:
        """Return True if text is a valid prefix of any word in the trie."""
        node = self.root
        for char in text:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

    def is_complete_word(self, text: str) -> bool:
        """Return True if text exactly matches a fully inserted word."""
        node = self.root
        for char in text:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word
