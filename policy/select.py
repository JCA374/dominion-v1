"""Command line for the model selector (the code is in policy/selector.py).

    python -m policy.select Cellar Chapel Moat Village Smithy Militia Market Witch Mine Artisan
    python -m policy.select --best-kingdom
"""
import sys

from policy.selector import main

if __name__ == "__main__":
    sys.exit(main())
