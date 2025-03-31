from abc import ABC, abstractmethod

import numpy as np
import torch


class BaseTemperatureSchedule(ABC):
    
    def __init__(self):
        self.step = 0
        
    def __call__(self, step=None):
        if step is None:
            step = self.step
        return self.get_temperature(step)
    
    @abstractmethod
    def get_temperature(self, step):
        return 


class ConstantTemperatureSchedule(BaseTemperatureSchedule):
    def __init__(self, temp):
        self.temp = temp
        super().__init__()

    def get_temperature(self, step):
        return self.temp

class LinearTemperatureSchedule(BaseTemperatureSchedule):
    def __init__(self, start_step, end_step, start_temp, end_temp):
        self.start_step = start_step
        self.end_step = end_step
        self.start_temp = start_temp
        self.end_temp = end_temp
        super().__init__()

    def get_temperature(self, step):
        """
        Linear temperature schedule from start_temp to end_temp over the course of end_step steps.
        """
        if step > self.end_step:
            return self.end_temp
        elif step < self.start_step:
            return self.start_temp
        else:
            return self.start_temp + (self.end_temp - self.start_temp) * (step / self.end_step)

