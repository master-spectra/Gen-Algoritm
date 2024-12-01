from abc import ABC, abstractmethod
from typing import Dict, Any, TypeVar, Generic, Optional, List
from dataclasses import dataclass
import random
import numpy as np
from copy import deepcopy

T = TypeVar('T')

class Gene(Generic[T]):
    """Базовый класс для гена"""
    def __init__(self,
                 value: T,
                 min_value: T,
                 max_value: T,
                 name: str = "",
                 metadata: Dict[str, Any] = None):
        self.value = value
        self.min_value = min_value
        self.max_value = max_value
        self.name = name
        self.metadata = metadata or {}

    def mutate(self, mutation_rate: float) -> None:
        """Абстрактный метод мутации"""
        pass

    def copy(self) -> 'Gene[T]':
        """Создание копии гена"""
        return deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            'value': self.value,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'name': self.name,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Gene[T]':
        """Создание гена из словаря"""
        return cls(
            value=data['value'],
            min_value=data['min_value'],
            max_value=data['max_value'],
            name=data.get('name', ""),
            metadata=data.get('metadata', {})
        )

class FloatGene(Gene[float]):
    """Реализация гена для вещественных чисел"""

    def __init__(self,
                 min_value: float,
                 max_value: float,
                 mutation_scale: float = 0.1,
                 name: str = "",
                 metadata: Dict[str, Any] = None):
        value = random.uniform(min_value, max_value)
        super().__init__(value, min_value, max_value, name, metadata)
        self.mutation_scale = mutation_scale

    def mutate(self, mutation_rate: float) -> None:
        """Мутация вещественного гена"""
        if random.random() < mutation_rate:
            range_size = self.max_value - self.min_value
            mutation = random.gauss(0, self.mutation_scale * range_size)
            self.value = np.clip(
                self.value + mutation,
                self.min_value,
                self.max_value
            )

class IntegerGene(Gene[int]):
    """Реализация гена для целых чисел"""

    def __init__(self,
                 min_value: int,
                 max_value: int,
                 mutation_step: int = 1,
                 name: str = "",
                 metadata: Dict[str, Any] = None):
        value = random.randint(min_value, max_value)
        super().__init__(value, min_value, max_value, name, metadata)
        self.mutation_step = mutation_step

    def mutate(self, mutation_rate: float) -> None:
        """Мутация целочисленного гена"""
        if random.random() < mutation_rate:
            # Определяем размер шага мутации
            range_size = self.max_value - self.min_value
            max_step = min(self.mutation_step, range_size // 2)
            step = random.randint(-max_step, max_step)

            # Применяем мутацию с ограничением на диапазон значений
            new_value = self.value + step
            self.value = min(max(new_value, self.min_value), self.max_value)

class NumericGene(Gene[float]):
    """Реализация гена для числовых значений"""

    def __init__(self,
                 value: float,
                 min_value: float,
                 max_value: float,
                 mutation_scale: float = 0.1,
                 **kwargs):
        super().__init__(value, min_value, max_value, **kwargs)
        self.mutation_scale = mutation_scale

    def mutate(self, mutation_rate: float) -> None:
        """Мутация числового гена"""
        if random.random() < mutation_rate:
            range_size = self.max_value - self.min_value
            mutation = random.gauss(0, self.mutation_scale * range_size)
            self.value = np.clip(
                self.value + mutation,
                self.min_value,
                self.max_value
            )

class BinaryGene(Gene[bool]):
    """Реализация гена для булевых значений"""

    def mutate(self, mutation_rate: float) -> None:
        """Мутация булевого гена"""
        if random.random() < mutation_rate:
            self.value = not self.value

class DiscreteGene(Gene[T]):
    """Реализация гена для дискретных значений"""

    def __init__(self,
                 value: T,
                 possible_values: List[T],
                 **kwargs):
        super().__init__(
            value=value,
            min_value=possible_values[0],
            max_value=possible_values[-1],
            **kwargs
        )
        self.possible_values = possible_values

    def mutate(self, mutation_rate: float) -> None:
        """Мутация дискретного гена"""
        if random.random() < mutation_rate:
            self.value = random.choice(self.possible_values)

class Chromosome(ABC):
    """Базовый класс для хромосомы в генетическом алгоритме"""

    def __init__(self):
        self.genes: Dict[str, Gene] = {}
        self.fitness: float = 0.0
        self.metadata: Dict[str, Any] = {}
        self.age: int = 0
        self.parent_ids: List[int] = []
        self._id: Optional[int] = None

    @property
    def id(self) -> int:
        """Уникальный идентификатор хромосомы"""
        if self._id is None:
            self._id = id(self)
        return self._id

    @abstractmethod
    def init_random(self) -> None:
        """Инициализация случайными значениями"""
        pass

    @abstractmethod
    def crossover(self, other: 'Chromosome') -> tuple['Chromosome', 'Chromosome']:
        """Скрещивание с другой хромосомой"""
        pass

    def mutate(self, mutation_rate: float) -> None:
        """Мутация генов"""
        for gene in self.genes.values():
            gene.mutate(mutation_rate)
        self.age += 1

    def copy(self) -> 'Chromosome':
        """Создание глубокой копии хромосомы"""
        new_chromosome = self.__class__()
        new_chromosome.genes = {
            name: gene.copy() for name, gene in self.genes.items()
        }
        new_chromosome.fitness = self.fitness
        new_chromosome.metadata = deepcopy(self.metadata)
        new_chromosome.age = self.age
        new_chromosome.parent_ids = self.parent_ids.copy()
        return new_chromosome

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            'genes': {
                name: gene.to_dict() for name, gene in self.genes.items()
            },
            'fitness': self.fitness,
            'metadata': self.metadata,
            'age': self.age,
            'parent_ids': self.parent_ids,
            'id': self.id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Chromosome':
        """Создание хромосомы из словаря"""
        chromosome = cls()
        for name, gene_data in data['genes'].items():
            gene_type = gene_data.get('type', 'numeric')
            if gene_type == 'numeric':
                gene = NumericGene.from_dict(gene_data)
            elif gene_type == 'binary':
                gene = BinaryGene.from_dict(gene_data)
            elif gene_type == 'discrete':
                gene = DiscreteGene.from_dict(gene_data)
            else:
                raise ValueError(f"Unknown gene type: {gene_type}")
            chromosome.genes[name] = gene

        chromosome.fitness = data['fitness']
        chromosome.metadata = data['metadata']
        chromosome.age = data['age']
        chromosome.parent_ids = data['parent_ids']
        chromosome._id = data['id']
        return chromosome

    def get_gene_value(self, name: str) -> Any:
        """Получение значения гена по имени"""
        return self.genes[name].value

    def set_gene_value(self, name: str, value: Any) -> None:
        """Установка значения гена по имени"""
        self.genes[name].value = value

    def get_genome(self) -> Dict[str, Any]:
        """Получение всех значений генов"""
        return {name: gene.value for name, gene in self.genes.items()}

    def calculate_distance(self, other: 'Chromosome') -> float:
        """Вычисление расстояния до другой хромосомы"""
        if set(self.genes.keys()) != set(other.genes.keys()):
            raise ValueError("Chromosomes have different gene sets")

        distances = []
        for name, gene in self.genes.items():
            other_gene = other.genes[name]
            if isinstance(gene, NumericGene):
                # Нормализованное евклидово расстояние для числовых генов
                range_size = gene.max_value - gene.min_value
                if range_size > 0:
                    dist = abs(gene.value - other_gene.value) / range_size
                    distances.append(dist)
            elif isinstance(gene, BinaryGene):
                # Расстояние Хэмминга для булевых генов
                distances.append(float(gene.value != other_gene.value))
            elif isinstance(gene, DiscreteGene):
                # Нормализованное позиционное расстояние для дискретных генов
                pos1 = gene.possible_values.index(gene.value)
                pos2 = gene.possible_values.index(other_gene.value)
                max_dist = len(gene.possible_values) - 1
                if max_dist > 0:
                    dist = abs(pos1 - pos2) / max_dist
                    distances.append(dist)

        return np.mean(distances) if distances else 0.0
