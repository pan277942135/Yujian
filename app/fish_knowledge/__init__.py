"""Fish Knowledge Database ORM models.

The public API and admin workflows live in this package as separate modules,
while the stable species identity remains aligned with ``species_catalog``.
"""

from app.fish_knowledge.fishing import FishFishing
from app.fish_knowledge.gallery import FishGalleryImage
from app.fish_knowledge.cards import FishCard
from app.fish_knowledge.profile import FishProfile
from app.fish_knowledge.ranking import FishRanking
from app.fish_knowledge.similarity import FishSimilarity
from app.fish_knowledge.species import FishSpecies
from app.fish_knowledge.video import FishVideo

__all__ = [
    "FishCard",
    "FishFishing",
    "FishGalleryImage",
    "FishProfile",
    "FishRanking",
    "FishSimilarity",
    "FishSpecies",
    "FishVideo",
]
