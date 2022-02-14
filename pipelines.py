from abc import ABC, abstractmethod


class Step(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def run(self):
        raise NotImplementedError


class Pipeline(ABC):

    def __init__(self):
        self.steps = []

    def register(self, step: Step):
        self.steps.append(step)

    def run(self):
        for step in self.steps:
            step.run()
