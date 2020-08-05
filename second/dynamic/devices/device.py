from easydict import EasyDict as edict

class Device:
    """
    A device is an object that produces an output sequence stream of data (self.stream).
    self.stream contains oldest-newest data from left to right.
    self.stream has a "capacity" -- if the stream is full, it will drop the oldest data.
    """
    def __init__(self, env, capacity=1):
        """
        Args:
            env: (simpy.Environment) simulation environment.
            stream_capacity: (int) maximum capacity of stream.
        """
        self.env = env
        self.stream = []
        self.capacity = capacity
    
    def publish(self, data):
        item = edict(timestamp=self.env.now, data=data)
        if len(self.stream) == self.capacity:
            self.stream.pop(0)
        self.stream.append(item)
    
    def reset(self, env):
        self.env = env
        self.stream.clear()
    
    def __str__(self):
        return f"{type(self)}\nstream(capacity={self.capacity}): {self.stream}"
