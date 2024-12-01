from abc import ABC, abstractmethod
from typing import List, TypeVar, Generic, Callable
import random
import numpy as np
from ..core.chromosome import Chromosome

C = TypeVar('C', bound=Chromosome)

class SelectionOperator(Generic[C], ABC):
    """Базовый класс для операторов селекции"""

    @abstractmethod
    def select(self, population: List[C], num_parents: int = 2) -> List[C]:
        """Выбор родителей из популяции"""
        pass

class TournamentSelection(SelectionOperator[C]):
    """Турнирная селекция"""

    def __init__(self, tournament_size: int = 3):
        self.tournament_size = tournament_size

    def select(self, population: List[C], num_parents: int = 2) -> List[C]:
        parents = []
        for _ in range(num_parents):
            tournament = random.sample(
                population,
                min(self.tournament_size, len(population))
            )
            winner = max(tournament, key=lambda x: x.fitness)
            parents.append(winner)
        return parents

class RouletteWheelSelection(SelectionOperator[C]):
    """Селекция методом рулетки"""

    def select(self, population: List[C], num_parents: int = 2) -> List[C]:
        total_fitness = sum(ind.fitness for ind in population)
        if total_fitness == 0:
            return random.sample(population, num_parents)

        parents = []
        for _ in range(num_parents):
            pick = random.uniform(0, total_fitness)
            current = 0
            for individual in population:
                current += individual.fitness
                if current > pick:
                    parents.append(individual)
                    break
            else:
                parents.append(population[-1])
        return parents

class RankSelection(SelectionOperator[C]):
    """Ранговая селекция"""

    def select(self, population: List[C], num_parents: int = 2) -> List[C]:
        # Сортируем по фитнесу и присваиваем ранги
        sorted_pop = sorted(population, key=lambda x: x.fitness)
        ranks = range(1, len(sorted_pop) + 1)
        total_rank = sum(ranks)

        parents = []
        for _ in range(num_parents):
            pick = random.uniform(0, total_rank)
            current = 0
            for rank, individual in zip(ranks, sorted_pop):
                current += rank
                if current > pick:
                    parents.append(individual)
                    break
            else:
                parents.append(sorted_pop[-1])
        return parents

class StochasticUniversalSampling(SelectionOperator[C]):
    """Стохастическая универсальная выборка"""

    def select(self, population: List[C], num_parents: int = 2) -> List[C]:
        total_fitness = sum(ind.fitness for ind in population)
        if total_fitness == 0:
            return random.sample(population, num_parents)

        distance = total_fitness / num_parents
        start = random.uniform(0, distance)
        points = [start + i * distance for i in range(num_parents)]

        parents = []
        for point in points:
            current = 0
            for individual in population:
                current += individual.fitness
                if current > point:
                    parents.append(individual)
                    break
            else:
                parents.append(population[-1])
        return parents

class AdaptiveSelection(SelectionOperator[C]):
    """Адаптивная селекция"""

    def __init__(self,
                 selection_operators: List[SelectionOperator[C]],
                 success_history_size: int = 10):
        self.operators = selection_operators
        self.success_rates = [1.0] * len(selection_operators)
        self.history_size = success_history_size
        self.history = []

    def select(self, population: List[C], num_parents: int = 2) -> List[C]:
        # Выбираем оператор на основе успешности
        total_rate = sum(self.success_rates)
        if total_rate == 0:
            operator = random.choice(self.operators)
        else:
            weights = [rate / total_rate for rate in self.success_rates]
            operator = random.choices(self.operators, weights=weights)[0]

        parents = operator.select(population, num_parents)
        self.history.append((operator, parents))

        if len(self.history) > self.history_size:
            self.history.pop(0)

        return parents

    def update_success_rates(self, fitness_improvement: float) -> None:
        """Обновление успешности операторов"""
        if not self.history:
            return

        operator, _ = self.history[-1]
        operator_index = self.operators.index(operator)

        # Обновляем успешность на основе улучшения фитнеса
        if fitness_improvement > 0:
            self.success_rates[operator_index] *= 1.1
        else:
            self.success_rates[operator_index] *= 0.9
