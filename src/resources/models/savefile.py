from dataclasses import dataclass
from typing import Any

from src.commands.command import OptimizationTarget

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

  def select(self, target: str | OptimizationTarget) -> float:
    if target in self.values: return self.values[target]
    return self.__dict__[target]

@dataclass
class SaveMeta(object):
  command: str
  arguments: dict

@dataclass
class SaveRecord(object):
  name: str
  meta: SaveMeta
  population: list[Individual]
  history: list[HistoryRecord]

  def select(self, target: OptimizationTarget) -> list[float]:
    return [record.select(target) for record in self.history]
