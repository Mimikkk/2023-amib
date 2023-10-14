from dataclasses import dataclass

from commands.command import OptimizationTarget

@dataclass
class Individual(object):
  _classname: str
  genotype: str
  values: dict[OptimizationTarget, float]

@dataclass
class Statistic(object):
  avg: float
  stddev: float
  min: float
  max: float

@dataclass
class HistoryRecord(object):
  gen: int
  nevals: int
  avg: float
  stddev: float
  min: float
  max: float
  values: list[dict[OptimizationTarget, float]]

@dataclass
class SaveFile(object):
  name: str
  population: list[Individual]
  history: list[HistoryRecord]
