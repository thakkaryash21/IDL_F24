import numpy as np

"""
NOTE: Do NOT change anything in the GradientBuffer class. We are providing it to you. 
As discussed in the writeup, we track two types of gradients in different ways.
This class only tracks the gradients for the input data.
For network parameters, their gradients are tracked as a class attribute, e.g. dW.
"""


class GradientBuffer:
    def __init__(self):
        # maps from the address of an array to its gradient
        self.memory = dict()

    @staticmethod
    def get_memory_loc(np_array):
        return np_array.__array_interface__["data"][0]

    def is_in_memory(self, np_array):
        return self.get_memory_loc(np_array) in self.memory

    def add_spot(self, np_array):        
        if not self.is_in_memory(np_array):
            self.memory[self.get_memory_loc(np_array)] = np.zeros(np_array.shape)
        else:
            assert (
                self.memory[self.get_memory_loc(np_array)].shape == np_array.shape
            ), "You cannot add the same array with different views to the buffer."

    def update_param(self, np_array, gradient):
        # If a constant then no gradient is propagated (ex. mask in ReLU or labels in loss)
        if type(gradient).__name__ == "NoneType":
            pass
        elif self.is_in_memory(np_array):
            self.memory[self.get_memory_loc(np_array)] += gradient
        else:
            raise Exception(
                "Attempted to add gradient for a variable not in gradient buffer."
            )

    def get_param(self, np_array):
        if self.is_in_memory(np_array):
            return self.memory[self.get_memory_loc(np_array)]
        else:
            print(f"unfound array: {np_array}")
            raise Exception(
                "Attempted to get gradient for a variable not in gradient buffer."
            )

    def set_param(self, np_array, gradient):
        if self.is_in_memory(np_array):
            self.memory[self.get_memory_loc(np_array)] = gradient
        else:
            raise Exception(
                "Attempted to set gradient for a variable not in gradient buffer."
            )

    def clear(self):
        self.memory = dict()
