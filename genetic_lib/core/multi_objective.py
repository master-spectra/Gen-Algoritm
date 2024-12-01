from typing import List, Set, Dict, TypeVar, Generic
import numpy as np
from .chromosome import Chromosome
from .fitness import MultiFitness

C = TypeVar('C', bound=Chromosome)

class ParetoFront(Generic[C]):
    """Класс для работы с фронтом Парето"""

    def __init__(self):
        self.solutions: List[C] = []
        self._objectives_range = None

    def update(self, population: List[C]) -> None:
        """Обновление фронта Парето новыми решениями"""
        # Добавляем новые решения
        candidates = self.solutions + population

        # Находим недоминируемые решения
        non_dominated = []
        for candidate in candidates:
            if not any(other.fitness.dominates(candidate.fitness)
                      for other in candidates if other != candidate):
                non_dominated.append(candidate)

        self.solutions = non_dominated
        self._objectives_range = None

    def get_normalized_objectives(self) -> np.ndarray:
        """Получение нормализованных значений целевых функций"""
        if not self.solutions:
            return np.array([])

        objectives = np.array([s.fitness.values for s in self.solutions])
        if self._objectives_range is None:
            self._objectives_range = {
                'min': np.min(objectives, axis=0),
                'max': np.max(objectives, axis=0)
            }

        denominator = (self._objectives_range['max'] - self._objectives_range['min'])
        denominator[denominator == 0] = 1  # Избегаем деления на ноль

        return (objectives - self._objectives_range['min']) / denominator

class NonDominatedSorting:
    """Реализация сортировки недоминируемых решений (NSGA-II)"""

    @staticmethod
    def fast_non_dominated_sort(population: List[C]) -> List[List[C]]:
        """Быстрая сортировка недоминируемых решений"""
        fronts: List[List[C]] = [[]]
        dominated_solutions: Dict[C, Set[C]] = {p: set() for p in population}
        domination_counts: Dict[C, int] = {p: 0 for p in population}

        # Для каждой пары решений определяем отношения доминирования
        for p in population:
            for q in population:
                if p.fitness.dominates(q.fitness):
                    dominated_solutions[p].add(q)
                elif q.fitness.dominates(p.fitness):
                    domination_counts[p] += 1

            if domination_counts[p] == 0:
                fronts[0].append(p)

        # Формируем остальные фронты
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in dominated_solutions[p]:
                    domination_counts[q] -= 1
                    if domination_counts[q] == 0:
                        next_front.append(q)
            i += 1
            if next_front:
                fronts.append(next_front)

        return fronts

    @staticmethod
    def calculate_crowding_distance(front: List[C]) -> Dict[C, float]:
        """Вычисление crowding distance для решений в одном фронте"""
        if len(front) <= 2:
            return {ind: float('inf') for ind in front}

        distances = {ind: 0.0 for ind in front}
        objectives = len(front[0].fitness.values)

        for m in range(objectives):
            # Сортируем решения по m-й целевой функции
            front.sort(key=lambda x: x.fitness.values[m])

            # Бесконечное расстояние для граничных точек
            distances[front[0]] = float('inf')
            distances[front[-1]] = float('inf')

            # Вычисляем расстояния для остальных точек
            obj_range = front[-1].fitness.values[m] - front[0].fitness.values[m]
            if obj_range == 0:
                continue

            for i in range(1, len(front) - 1):
                distances[front[i]] += (
                    front[i + 1].fitness.values[m] - front[i - 1].fitness.values[m]
                ) / obj_range

        return distances

class CrowdedComparisonOperator:
    """Оператор сравнения с учетом скученности (NSGA-II)"""

    @staticmethod
    def compare(ind1: C, ind2: C, rank1: int, rank2: int,
                crowding_distances: Dict[C, float]) -> int:
        """
        Сравнение двух решений по рангу и crowding distance
        Возвращает:
            -1 если ind1 лучше ind2
            1 если ind2 лучше ind1
            0 если решения равноценны
        """
        if rank1 < rank2:
            return -1
        if rank1 > rank2:
            return 1
        if crowding_distances[ind1] > crowding_distances[ind2]:
            return -1
        if crowding_distances[ind1] < crowding_distances[ind2]:
            return 1
        return 0
