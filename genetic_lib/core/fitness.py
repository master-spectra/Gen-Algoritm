from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np

class MultiFitness:
    """Класс для хранения результатов многокритериальной оценки"""
    def __init__(self, values: List[float], weights: List[float] = None):
        self.values = np.array(values)
        self.weights = np.array(weights) if weights else np.ones(len(values))
        self._aggregate = None

    @property
    def aggregate(self) -> float:
        """Агрегированное значение фитнеса"""
        if self._aggregate is None:
            self._aggregate = np.sum(self.values * self.weights) / np.sum(self.weights)
        return self._aggregate

    def dominates(self, other: 'MultiFitness') -> bool:
        """Проверка доминирования по Парето"""
        return (np.all(self.values >= other.values) and
                np.any(self.values > other.values))

class FitnessFunction(ABC):
    """Базовый класс для фитнес-функций"""

    def __init__(self, objectives: List[str], weights: List[float] = None):
        """
        Args:
            objectives: Список названий целевых функций
            weights: Веса для каждой целевой функции
        """
        self.objectives = objectives
        self.weights = weights if weights else [1.0] * len(objectives)
        self.metadata: Dict[str, Any] = {}

    @abstractmethod
    async def evaluate(self, chromosome: 'Chromosome') -> MultiFitness:
        """Асинхронная оценка хромосомы"""
        pass

    def add_metadata(self, key: str, value: Any) -> None:
        """Добавление метаданных"""
        self.metadata[key] = value

    def get_metadata(self, key: str) -> Any:
        """Получение метаданных"""
        return self.metadata.get(key)

class WeightedSumFitness(FitnessFunction):
    """Пример реализации взвешенной суммы целевых функций"""

    async def evaluate(self, chromosome: 'Chromosome') -> MultiFitness:
        values = []
        for objective in self.objectives:
            # Получаем значение для каждой целевой функции
            value = await self._evaluate_objective(chromosome, objective)
            values.append(value)

        return MultiFitness(values, self.weights)

    async def _evaluate_objective(self, chromosome: 'Chromosome',
                                objective: str) -> float:
        """Оценка отдельной целевой функции"""
        # Этот метод должен быть переопределен в конкретных реализациях
        raise NotImplementedError
