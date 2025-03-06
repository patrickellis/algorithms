
"""
├── setup.py
└── tests
    ├── __pycache__
    │   └── test_octodns_provider_edgedns.cpython-310-pytest-7.4.3.pyc
    ├── config
    │   └── unit.tests.yaml
    ├── fixtures
    │   ├── edgedns-invalid-content.json
    │   ├── edgedns-records-prev-other.json
    │   ├── edgedns-records-prev.json
    │   └── edgedns-records.json
    └── test_octodns_provider_edgedns.py
"""
from pathlib import Path

def generate_tree(path: Path, prefix=""):
    """
    Recursively generates a tree-like structure for a directory using pathlib.

    Args:
        path (Path): The directory to generate the tree for.
        prefix (str): The prefix used for tree formatting (default is empty).

    Returns:
        None
    """
    try:
        entries = list(path.iterdir())
    except PermissionError:
        print(f"{prefix}[Permission Denied]")
        return
    except FileNotFoundError:
        print(f"{prefix}[Not Found]")
        return

    entries.sort(key=lambda entry: entry.name.lower())  # Sort entries alphabetically
    entries_count = len(entries)

    for index, entry in enumerate(entries):
        connector = "└── " if index == entries_count - 1 else "├── "

        print(f"{prefix}{connector}{entry.name}")

        if entry.is_dir():
            # Use appropriate prefix for the next level
            new_prefix = prefix + ("    " if index == entries_count - 1 else "│   ")
            generate_tree(entry, new_prefix)


# Example usage
if __name__ == "__main__":
    starting_path = Path("/Users/pes28/Sky/octodns-edgedns/")
    print(starting_path)
    generate_tree(starting_path)

