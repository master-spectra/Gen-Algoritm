from typing import List, TypeVar, Generic, Dict, Any
import numpy as np
from .chromosome import Chromosome

C = TypeVar('C', bound=Chromosome)

class ParetoFront(Generic[C]):
    """Класс для работы с фронтом Парето"""

    def __init__(self):
        self.solutions: List[C] = []
        self._objectives_range = None

    def update(self, population: List[C]) -> None:
        """
        Обновление фронта Парето новыми решениями

        Args:
            population: Список хромосом для обновления фронта
        """
        # Добавляем новые решения
        candidates = self.solutions + population

        # Находим недоминируемые решения
        non_dominated = []
        for candidate in candidates:
            if not any(self._dominates(other.fitness.values, candidate.fitness.values)
                      for other in candidates if other != candidate):
                non_dominated.append(candidate)

        self.solutions = non_dominated
        self._objectives_range = None

    def get_normalized_objectives(self) -> np.ndarray:
        """
        Получение нормализованных значений целевых функций

        Returns:
            Массив нормализованных значений целевых функций
        """
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

    def calculate_hypervolume(self, reference_point: np.ndarray = None) -> float:
        """
        Вычисление гиперобъема фронта Парето

        Args:
            reference_point: Референсная точка для вычисления гиперобъема

        Returns:
            Значение гиперобъема
        """
        if not self.solutions:
            return 0.0

        objectives = self.get_normalized_objectives()
        if reference_point is None:
            reference_point = np.ones(objectives.shape[1])

        volume = 0.0
        sorted_points = objectives[objectives[:, 0].argsort()]

        if len(sorted_points) == 1:
            volume = np.prod(reference_point - sorted_points[0])
        else:
            for i in range(len(sorted_points) - 1):
                point = sorted_points[i]
                next_point = sorted_points[i + 1]

                # Вычисляем объем слоя между точками
                layer_volume = (next_point[0] - point[0]) * np.prod(
                    reference_point[1:] - point[1:]
                )
                volume += layer_volume

            # Добавляем последний слой
            last_point = sorted_points[-1]
            last_volume = np.prod(reference_point - last_point)
            volume += last_volume

        return volume

    @staticmethod
    def _dominates(values1: np.ndarray, values2: np.ndarray) -> bool:
        """
        Проверка доминирования одного решения над другим

        Args:
            values1: Значения целевых функций первого решения
            values2: Значения целевых функций второго решения

        Returns:
            True если первое решение доминирует над вторым
        """
        return (np.all(values1 >= values2) and
                np.any(values1 > values2))

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            'solutions': [s.to_dict() for s in self.solutions],
            'objectives_range': self._objectives_range
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], chromosome_class: type) -> 'ParetoFront':
        """Создание из словаря"""
        front = cls()
        front.solutions = [
            chromosome_class.from_dict(s_data)
            for s_data in data['solutions']
        ]
        front._objectives_range = data['objectives_range']
        return front
