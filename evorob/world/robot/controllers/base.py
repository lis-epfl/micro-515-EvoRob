from typing import Any

from abc import ABC, abstractmethod


class Controller(ABC):

    @abstractmethod
    def get_action(self, state) -> Any:
        return NotImplementedError

    def geno2pheno(self, genotype) -> None:
        return NotImplementedError