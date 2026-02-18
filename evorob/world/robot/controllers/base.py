from typing import Any

from abc import ABC, abstractmethod


class Controller(ABC):

    @abstractmethod
    def get_action(self, state) -> Any:
        raise NotImplementedError

    def reset_controller(self, batch_size=1) -> None:
        ...

    def geno2pheno(self, genotype) -> None:
        raise NotImplementedError
