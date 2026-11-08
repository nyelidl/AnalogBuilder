"""Run PLIP with NumPy 2 compatibility, without modifying installed packages."""
from __future__ import annotations


def int32_to_negative(value):
    """Preserve PLIP 2.3's -1000..-1 residue-number workaround."""
    number = int(value)
    return number - 2**32 if 2**32 - 1000 <= number <= 2**32 - 1 else number


def main():
    # Patch before importing modules that bind this function via `from ...`.
    import plip.basic.supplemental as supplemental
    supplemental.int32_to_negative = int32_to_negative
    from plip.plipcmd import main as plip_main
    plip_main()


if __name__ == "__main__":
    main()
