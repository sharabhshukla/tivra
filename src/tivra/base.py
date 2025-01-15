from abc import ABCMeta, abstractmethod


class Extractor(metaclass=ABCMeta):

    @abstractmethod
    def no_constraints(self):
        pass

    @abstractmethod
    def no_vars(self):
        pass

    @abstractmethod
    def extract_all(self):
        pass


