"""
Catppuccin theme setup for matplotlib / seaborn.

Usage:
    from theme import apply, C, PALETTE

    apply()               # Mocha by default
    apply("latte")        # or latte / frappe / macchiato

    plt.bar(..., color=C.blue)
    plt.bar(..., color=C.peach)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import catppuccin

# Available flavours
FLAVOURS = ("latte", "frappe", "macchiato", "mocha")


class Colors:
    """Dot-accessible hex colours for a given Catppuccin flavour."""

    def __init__(self, flavour: str = "mocha"):
        palette = getattr(catppuccin.PALETTE, flavour).colors
        self.rosewater = palette.rosewater.hex
        self.flamingo  = palette.flamingo.hex
        self.pink      = palette.pink.hex
        self.mauve     = palette.mauve.hex
        self.red       = palette.red.hex
        self.maroon    = palette.maroon.hex
        self.peach     = palette.peach.hex
        self.yellow    = palette.yellow.hex
        self.green     = palette.green.hex
        self.teal      = palette.teal.hex
        self.sky       = palette.sky.hex
        self.sapphire  = palette.sapphire.hex
        self.blue      = palette.blue.hex
        self.lavender  = palette.lavender.hex
        self.text      = palette.text.hex
        self.subtext1  = palette.subtext1.hex
        self.overlay0  = palette.overlay0.hex
        self.surface2  = palette.surface2.hex
        self.base      = palette.base.hex
        self.mantle    = palette.mantle.hex
        self.crust     = palette.crust.hex

    # Handy ordered list for cycling through plots
    @property
    def cycle(self):
        return [
            self.blue, self.peach, self.green, self.mauve,
            self.red, self.teal, self.yellow, self.flamingo,
        ]


def apply(flavour: str = "mocha") -> Colors:
    """Apply Catppuccin theme globally and return a Colors instance."""
    if flavour not in FLAVOURS:
        raise ValueError(f"flavour must be one of {FLAVOURS}")

    c = Colors(flavour)

    # seaborn first, catppuccin last — seaborn resets figure/axes facecolor
    # so it must run before plt.style.use(), not after
    sns.set_theme(style="darkgrid")
    plt.style.use(getattr(catppuccin.PALETTE, flavour).identifier)
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=c.cycle)
    return c


# Default
PALETTE = catppuccin.PALETTE
