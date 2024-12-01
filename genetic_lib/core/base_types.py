"""Базовые типы для генетического алгоритма"""
from typing import TypeVar
from .chromosome import Chromosome, FloatGene

# Определяем базовый тип для хромосом
C = TypeVar('C', bound=Chromosome)

# Определяем базовые классы для разных типов хромосом
class GameChromosome(Chromosome):
    """Базовый класс для игровых хромосом"""
    def __init__(self):
        super().__init__()
        self.genes = {}

class OptimizationChromosome(Chromosome):
    """Базовый класс для оптимизационных хромосом"""
    def __init__(self):
        super().__init__()
        self.genes = {}
