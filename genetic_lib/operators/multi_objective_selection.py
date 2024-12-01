from typing import List, Dict, TypeVar
from ..core.chromosome import Chromosome
from ..core.multi_objective import NonDominatedSorting, CrowdedComparisonOperator
from .selection import SelectionOperator

C = TypeVar('C', bound=Chromosome)

class NSGAIISelection(SelectionOperator[C]):
    """Оператор селекции на основе NSGA-II"""

    def __init__(self):
        self.sorter = NonDominatedSorting()
        self.comparator = CrowdedComparisonOperator()

    def select(self, population: List[C], num_parents: int = 2) -> List[C]:
        """
        Выбор родителей с использованием недоминируемой сортировки
        и crowding distance
        """
        # Получаем фронты
        fronts = self.sorter.fast_non_dominated_sort(population)

        # Вычисляем ранги
        ranks = {}
        for rank, front in enumerate(fronts):
            for individual in front:
                ranks[individual] = rank

        # Вычисляем crowding distance для каждого фронта
        crowding_distances = {}
        for front in fronts:
            distances = self.sorter.calculate_crowding_distance(front)
            crowding_distances.update(distances)

        # Выбираем родителей с помощью турнирной селекции
        parents = []
        for _ in range(num_parents):
            # Выбираем двух случайных кандидатов
            candidates = self._tournament_selection(population, ranks,
                                                 crowding_distances)
            parents.append(candidates)

        return parents

    def _tournament_selection(self, population: List[C], ranks: Dict[C, int],
                            crowding_distances: Dict[C, float],
                            tournament_size: int = 2) -> C:
        """Турнирный отбор с учетом рангов и crowding distance"""
        import random

        # Выбираем случайных кандидатов
        candidates = random.sample(population, tournament_size)

        # Находим лучшего кандидата
        best = candidates[0]
        for candidate in candidates[1:]:
            comparison = self.comparator.compare(
                candidate, best,
                ranks[candidate], ranks[best],
                crowding_distances
            )
            if comparison == -1:  # candidate лучше best
                best = candidate

        return best
