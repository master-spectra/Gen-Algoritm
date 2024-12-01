from typing import List, TypeVar, Generic, Callable, Dict, Any, Optional
import random
from .chromosome import Chromosome

C = TypeVar('C', bound=Chromosome)

class Population(Generic[C]):
    """Класс, представляющий популяцию в генетическом алгоритме"""

    def __init__(self,
                 chromosomes: List[C],
                 max_size: int,
                 metadata: Dict[str, Any] = None):
        """
        Инициализация популяции

        Args:
            chromosomes: Список хромосом
            max_size: Максимальный размер популяции
            metadata: Дополнительные метаданные
        """
        self.chromosomes = chromosomes
        self.max_size = max_size
        self.metadata = metadata or {}
        self.generation = 0
        self._best_individual: Optional[C] = None
        self._worst_individual: Optional[C] = None
        self._mean_fitness: Optional[float] = None

    @property
    def size(self) -> int:
        """Текущий размер популяции"""
        return len(self.chromosomes)

    @property
    def best_individual(self) -> C:
        """Лучшая особь в популяции"""
        if self._best_individual is None:
            self._best_individual = max(
                self.chromosomes,
                key=lambda x: x.fitness
            )
        return self._best_individual

    @property
    def worst_individual(self) -> C:
        """Худшая особь в популяции"""
        if self._worst_individual is None:
            self._worst_individual = min(
                self.chromosomes,
                key=lambda x: x.fitness
            )
        return self._worst_individual

    @property
    def mean_fitness(self) -> float:
        """Среднее значение фитнеса в популяции"""
        if self._mean_fitness is None:
            self._mean_fitness = sum(
                c.fitness for c in self.chromosomes
            ) / len(self.chromosomes)
        return self._mean_fitness

    def add_individual(self, chromosome: C) -> None:
        """Добавление новой особи в популяцию"""
        self.chromosomes.append(chromosome)
        self._invalidate_cache()

    def remove_individual(self, chromosome: C) -> None:
        """Удаление особи из популяции"""
        self.chromosomes.remove(chromosome)
        self._invalidate_cache()

    def sort_by_fitness(self, reverse: bool = True) -> None:
        """Сортировка популяции по фитнесу"""
        self.chromosomes.sort(
            key=lambda x: x.fitness,
            reverse=reverse
        )

    def get_elite(self, n: int) -> List[C]:
        """Получение n лучших особей"""
        self.sort_by_fitness(reverse=True)
        return self.chromosomes[:n]

    def truncate(self) -> None:
        """Обрезка популяции до максимального размера"""
        if len(self.chromosomes) > self.max_size:
            self.sort_by_fitness(reverse=True)
            self.chromosomes = self.chromosomes[:self.max_size]
            self._invalidate_cache()

    def calculate_diversity(self) -> float:
        """Вычисление разнообразия популяции"""
        if not self.chromosomes:
            return 0.0

        total_distance = 0.0
        count = 0

        # Вычисляем среднее расстояние между всеми парами особей
        for i in range(len(self.chromosomes)):
            for j in range(i + 1, len(self.chromosomes)):
                total_distance += self.chromosomes[i].calculate_distance(
                    self.chromosomes[j]
                )
                count += 1

        return total_distance / count if count > 0 else 0.0

    def get_statistics(self) -> Dict[str, float]:
        """Получение статистики популяции"""
        return {
            'best_fitness': self.best_individual.fitness,
            'worst_fitness': self.worst_individual.fitness,
            'mean_fitness': self.mean_fitness,
            'diversity': self.calculate_diversity(),
            'size': self.size,
            'generation': self.generation
        }

    def _invalidate_cache(self) -> None:
        """Сброс кэшированных значений"""
        self._best_individual = None
        self._worst_individual = None
        self._mean_fitness = None

    @classmethod
    def create_random(cls,
                     size: int,
                     chromosome_factory: Callable[[], C],
                     metadata: Dict[str, Any] = None) -> 'Population[C]':
        """
        Создание случайной популяции

        Args:
            size: Размер популяции
            chromosome_factory: Фабричная функция для создания хромосом
            metadata: Дополнительные метаданные

        Returns:
            Новая популяция со случайными особями
        """
        chromosomes = []
        for _ in range(size):
            chromosome = chromosome_factory()
            chromosome.init_random()
            chromosomes.append(chromosome)

        return cls(chromosomes, size, metadata)

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование популяции в словарь"""
        return {
            'chromosomes': [c.to_dict() for c in self.chromosomes],
            'max_size': self.max_size,
            'metadata': self.metadata,
            'generation': self.generation
        }

    @classmethod
    def from_dict(cls,
                 data: Dict[str, Any],
                 chromosome_class: type) -> 'Population':
        """Создание популяции из словаря"""
        chromosomes = [
            chromosome_class.from_dict(c_data)
            for c_data in data['chromosomes']
        ]
        population = cls(
            chromosomes=chromosomes,
            max_size=data['max_size'],
            metadata=data['metadata']
        )
        population.generation = data['generation']
        return population

    def copy(self, deep: bool = True) -> 'Population':
        """
        Создает копию популяции

        Args:
            deep: Если True, создает глубокую копию всех хромосом
                  Если False, создает поверхностную копию

        Returns:
            Population: Копия популяции
        """
        if deep:
            copied_chromosomes = [chromosome.copy() for chromosome in self.chromosomes]
        else:
            copied_chromosomes = self.chromosomes.copy()

        copied_population = Population(
            chromosomes=copied_chromosomes,
            max_size=self.max_size,
            metadata=self.metadata.copy() if self.metadata else None
        )
        copied_population.generation = self.generation
        return copied_population
