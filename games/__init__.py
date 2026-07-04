from .registry import GameSpec, get_game, list_games, register, sanity_check
from . import chizat_example_4_1, bilinear, convex_concave, sc_sc  # noqa: F401  (registers games)

__all__ = ["GameSpec", "get_game", "list_games", "register", "sanity_check"]
