"""
Genetic Algorithm Library
A powerful and flexible library for genetic optimization with multi-objective support
"""

from .core import (
    Gene,
    NumericGene,
    FloatGene,
    IntegerGene,
    BinaryGene,
    DiscreteGene,
    Chromosome,
    Population,
    Evolution,
    MultiObjectiveEvolution,
    FitnessFunction,
    MultiFitness
)

from .operators import (
    SelectionOperator,
    TournamentSelection,
    RouletteWheelSelection,
    RankSelection,
    StochasticUniversalSampling,
    AdaptiveSelection,
    CrossoverOperator,
    SinglePointCrossover,
    TwoPointCrossover,
    UniformCrossover,
    AdaptiveCrossover,
    MutationOperator,
    GaussianMutation,
    UniformMutation,
    AdaptiveMutation,
    PolynomialMutation,
    NSGAIISelection
)

from .utils import (
    EvolutionVisualizer,
    StatePersistence,
    AutoCheckpoint
)

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = [
    # Core
    'Gene',
    'NumericGene',
    'FloatGene',
    'IntegerGene',
    'BinaryGene',
    'DiscreteGene',
    'Chromosome',
    'Population',
    'Evolution',
    'MultiObjectiveEvolution',
    'FitnessFunction',
    'MultiFitness',
    # Operators
    'SelectionOperator',
    'TournamentSelection',
    'RouletteWheelSelection',
    'RankSelection',
    'StochasticUniversalSampling',
    'AdaptiveSelection',
    'CrossoverOperator',
    'SinglePointCrossover',
    'TwoPointCrossover',
    'UniformCrossover',
    'AdaptiveCrossover',
    'MutationOperator',
    'GaussianMutation',
    'UniformMutation',
    'AdaptiveMutation',
    'PolynomialMutation',
    'NSGAIISelection',
    # Utils
    'EvolutionVisualizer',
    'StatePersistence',
    'AutoCheckpoint'
]
