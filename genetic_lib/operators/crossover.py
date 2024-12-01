from abc import ABC, abstractmethod
from typing import List, Tuple, TypeVar, Generic, Dict
import random
import numpy as np
from ..core.chromosome import Chromosome

C = TypeVar('C', bound=Chromosome)

class CrossoverOperator(Generic[C], ABC):
    """Базовый класс для операторов скрещивания"""

    @abstractmethod
    def crossover(self, parent1: C, parent2: C) -> Tuple[C, C]:
        """Скрещивание двух родителей"""
        pass

class SinglePointCrossover(CrossoverOperator[C]):
    """Одноточечное скрещивание"""

    def crossover(self, parent1: C, parent2: C) -> Tuple[C, C]:
        genes1 = parent1.to_dict()
        genes2 = parent2.to_dict()

        # Выбираем точку скрещивания
        point = random.randint(1, len(genes1) - 1)
        keys = list(genes1.keys())

        # Создаем новые наборы генов
        child1_genes = {
            k: genes1[k] if i < point else genes2[k]
            for i, k in enumerate(keys)
        }
        child2_genes = {
            k: genes2[k] if i < point else genes1[k]
            for i, k in enumerate(keys)
        }

        # Создаем потомков
        child1 = parent1.__class__()
        child2 = parent1.__class__()
        child1.from_dict(child1_genes)
        child2.from_dict(child2_genes)

        return child1, child2

class TwoPointCrossover(CrossoverOperator[C]):
    """Двухточечное скрещивание"""

    def crossover(self, parent1: C, parent2: C) -> Tuple[C, C]:
        genes1 = parent1.to_dict()
        genes2 = parent2.to_dict()

        # Выбираем две точки скрещивания
        point1, point2 = sorted(random.sample(range(1, len(genes1)), 2))
        keys = list(genes1.keys())

        # Создаем новые наборы генов
        child1_genes = {}
        child2_genes = {}

        for i, k in enumerate(keys):
            if i < point1 or i >= point2:
                child1_genes[k] = genes1[k]
                child2_genes[k] = genes2[k]
            else:
                child1_genes[k] = genes2[k]
                child2_genes[k] = genes1[k]

        # Создаем потомков
        child1 = parent1.__class__()
        child2 = parent1.__class__()
        child1.from_dict(child1_genes)
        child2.from_dict(child2_genes)

        return child1, child2

class UniformCrossover(CrossoverOperator[C]):
    """Равномерное скрещивание"""

    def __init__(self, swap_probability: float = 0.5):
        self.swap_probability = swap_probability

    def crossover(self, parent1: C, parent2: C) -> Tuple[C, C]:
        genes1 = parent1.to_dict()
        genes2 = parent2.to_dict()

        child1_genes = {}
        child2_genes = {}

        for key in genes1.keys():
            if random.random() < self.swap_probability:
                child1_genes[key] = genes2[key]
                child2_genes[key] = genes1[key]
            else:
                child1_genes[key] = genes1[key]
                child2_genes[key] = genes2[key]

        child1 = parent1.__class__()
        child2 = parent1.__class__()
        child1.from_dict(child1_genes)
        child2.from_dict(child2_genes)

        return child1, child2

class BlendCrossover(CrossoverOperator[C]):
    """Скрещивание смешиванием (для числовых генов)"""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def crossover(self, parent1: C, parent2: C) -> Tuple[C, C]:
        genes1 = parent1.to_dict()
        genes2 = parent2.to_dict()

        child1_genes = {}
        child2_genes = {}

        for key in genes1.keys():
            if isinstance(genes1[key], (int, float)):
                # Определяем границы смешивания
                min_val = min(genes1[key], genes2[key])
                max_val = max(genes1[key], genes2[key])
                range_val = max_val - min_val

                # Расширяем границы
                lower = min_val - self.alpha * range_val
                upper = max_val + self.alpha * range_val

                # Генерируем значения для потомков
                child1_genes[key] = random.uniform(lower, upper)
                child2_genes[key] = random.uniform(lower, upper)
            else:
                # Для нечисловых генов используем случайный выбор
                if random.random() < 0.5:
                    child1_genes[key] = genes1[key]
                    child2_genes[key] = genes2[key]
                else:
                    child1_genes[key] = genes2[key]
                    child2_genes[key] = genes1[key]

        child1 = parent1.__class__()
        child2 = parent1.__class__()
        child1.from_dict(child1_genes)
        child2.from_dict(child2_genes)

        return child1, child2

class AdaptiveCrossover(CrossoverOperator[C]):
    """Адаптивное скрещивание"""

    def __init__(self,
                 operators: List[CrossoverOperator[C]],
                 success_history_size: int = 10):
        self.operators = operators
        self.success_rates = [1.0] * len(operators)
        self.history_size = success_history_size
        self.history = []

    def crossover(self, parent1: C, parent2: C) -> Tuple[C, C]:
        # Выбираем оператор на основе успешности
        total_rate = sum(self.success_rates)
        if total_rate == 0:
            operator = random.choice(self.operators)
        else:
            weights = [rate / total_rate for rate in self.success_rates]
            operator = random.choices(self.operators, weights=weights)[0]

        children = operator.crossover(parent1, parent2)
        self.history.append((operator, children))

        if len(self.history) > self.history_size:
            self.history.pop(0)

        return children

    def update_success_rates(self, fitness_improvements: Tuple[float, float]) -> None:
        """Обновление успешности операторов"""
        if not self.history:
            return

        operator, _ = self.history[-1]
        operator_index = self.operators.index(operator)

        # Обновляем успешность на основе улучшения фитнеса потомков
        avg_improvement = sum(fitness_improvements) / len(fitness_improvements)
        if avg_improvement > 0:
            self.success_rates[operator_index] *= 1.1
        else:
            self.success_rates[operator_index] *= 0.9
