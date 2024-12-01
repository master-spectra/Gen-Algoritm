from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List
import random
import numpy as np
from ..core.chromosome import Chromosome, NumericGene, BinaryGene, DiscreteGene

C = TypeVar('C', bound=Chromosome)

class MutationOperator(Generic[C], ABC):
    """Базовый класс для операторов мутации"""

    @abstractmethod
    def mutate(self, chromosome: C, mutation_rate: float) -> C:
        """Мутация хромосомы"""
        pass

class PolynomialMutation(MutationOperator[C]):
    """
    Полиномиальная мутация для вещественных генов.
    Использует полиномиальное распределение для генерации мутаций.
    """

    def __init__(self, distribution_index: float = 20.0):
        """
        Args:
            distribution_index: Индекс распределения (η_m).
                              Большие значения создают мутации ближе к родителю.
        """
        self.distribution_index = distribution_index

    def mutate(self, chromosome: C, mutation_rate: float) -> C:
        """
        Применяет полиномиальную мутацию к хромосоме.

        Args:
            chromosome: Хромосома для мутации
            mutation_rate: Вероятность мутации для каждого гена

        Returns:
            Мутированная хромосома
        """
        mutated = chromosome.__class__()
        genes = chromosome.to_dict()

        for key, gene in genes.items():
            if isinstance(gene, NumericGene):
                if random.random() < mutation_rate:
                    # Полиномиальная мутация для числовых генов
                    delta_1 = (gene.value - gene.min_value) / (gene.max_value - gene.min_value)
                    delta_2 = (gene.max_value - gene.value) / (gene.max_value - gene.min_value)

                    # Генерация случайного числа
                    rand = random.random()
                    mut_pow = 1.0 / (self.distribution_index + 1.0)

                    if rand <= 0.5:
                        xy = 1.0 - delta_1
                        val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (self.distribution_index + 1.0))
                        delta_q = val ** mut_pow - 1.0
                    else:
                        xy = 1.0 - delta_2
                        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (self.distribution_index + 1.0))
                        delta_q = 1.0 - val ** mut_pow

                    # Вычисление нового значения
                    value = gene.value + delta_q * (gene.max_value - gene.min_value)
                    # Ограничение значения в допустимых пределах
                    value = np.clip(value, gene.min_value, gene.max_value)
                    gene.value = value
            else:
                gene.mutate(mutation_rate)

        mutated.from_dict(genes)
        return mutated

class GaussianMutation(MutationOperator[C]):
    """Gaussian mutation for real-valued genes"""

    def __init__(self, scale: float = 0.1):
        """
        Args:
            scale: Mutation scale (standard deviation)
        """
        self.scale = scale

    def mutate(self, chromosome: C, mutation_rate: float) -> C:
        """
        Apply Gaussian mutation to chromosome

        Args:
            chromosome: Chromosome to mutate
            mutation_rate: Mutation probability for each gene

        Returns:
            Mutated chromosome
        """
        mutated = chromosome.copy()

        for gene in mutated.genes.values():
            if isinstance(gene, NumericGene):
                if random.random() < mutation_rate:
                    # Gaussian mutation for numeric genes
                    range_size = gene.max_value - gene.min_value
                    mutation = random.gauss(0, self.scale * range_size)
                    value = gene.value + mutation
                    gene.value = np.clip(value, gene.min_value, gene.max_value)
            else:
                gene.mutate(mutation_rate)

        return mutated

class UniformMutation(MutationOperator[C]):
    """Равномерная мутация"""

    def mutate(self, chromosome: C, mutation_rate: float) -> C:
        mutated = chromosome.__class__()
        genes = chromosome.to_dict()

        for key, gene in genes.items():
            if random.random() < mutation_rate:
                if isinstance(gene, NumericGene):
                    # Равномерная мутация для числовых генов
                    gene.value = random.uniform(gene.min_value, gene.max_value)
                elif isinstance(gene, BinaryGene):
                    # Инверсия для булевых генов
                    gene.value = not gene.value
                elif isinstance(gene, DiscreteGene):
                    # Случайный выбор для дискретных генов
                    gene.value = random.choice(gene.possible_values)

        mutated.from_dict(genes)
        return mutated

class SwapMutation(MutationOperator[C]):
    """Мутация обменом значений между генами"""

    def mutate(self, chromosome: C, mutation_rate: float) -> C:
        mutated = chromosome.__class__()
        genes = chromosome.to_dict()
        gene_keys = list(genes.keys())

        if len(gene_keys) >= 2 and random.random() < mutation_rate:
            # Выбираем два случайных гена для обмена
            idx1, idx2 = random.sample(range(len(gene_keys)), 2)
            key1, key2 = gene_keys[idx1], gene_keys[idx2]

            # Обмениваем значения
            genes[key1].value, genes[key2].value = genes[key2].value, genes[key1].value

        mutated.from_dict(genes)
        return mutated

class AdaptiveMutation(MutationOperator[C]):
    """Адаптивная мутация, которая выбирает и применяет различные операторы мутации"""

    def __init__(self, operators: List[MutationOperator[C]], success_history_size: int = 10):
        self.operators = operators
        self.success_rates = [1.0] * len(operators)
        self.history_size = success_history_size
        self.history = []

    def mutate(self, chromosome: C, mutation_rate: float) -> C:
        # Выбираем оператор на основе успешности
        total_rate = sum(self.success_rates)
        if total_rate == 0:
            operator = random.choice(self.operators)
        else:
            weights = [rate / total_rate for rate in self.success_rates]
            operator = random.choices(self.operators, weights=weights)[0]

        mutated = operator.mutate(chromosome, mutation_rate)
        self.history.append((operator, mutated))

        if len(self.history) > self.history_size:
            self.history.pop(0)

        return mutated

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
