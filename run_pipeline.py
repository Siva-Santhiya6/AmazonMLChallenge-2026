import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

# Add code/business_entity_resolution to sys.path
repo_root = os.path.dirname(os.path.abspath(__file__))
pkg_path = os.path.join(repo_root, "code", "business_entity_resolution")
sys.path.insert(0, pkg_path)

from src.pipeline import main

if __name__ == "__main__":
    main()
