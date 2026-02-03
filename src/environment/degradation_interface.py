import abc

class DegradationProcess(abc.ABC):
    """
    Abstract interface for environment degradation processes.
    All degradation patterns must implement this interface.
    """
    @abc.abstractmethod
    def get_noise(self, step):
        pass

    @abc.abstractmethod
    def get_latency_load(self, step):
        pass
