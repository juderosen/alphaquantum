import gymnasium as gym # TODO: do we need gym? right now this is just here to satisfy muax

class StatefulWrapper(gym.Env):
    def __init__(self, stateless_env, *args, **kwargs):
        self.stateless_env = stateless_env(*args, **kwargs)
        self.state = self.stateless_env.reset()[0]

    def step(self, *args, **kwargs):
        results = self.stateless_env.step(self.state, *args, **kwargs)
        self.state = results[0]
        return results[1:]

    def reset(self, *args, **kwargs):
        results = self.stateless_env.reset(*args, **kwargs)
        self.state = results[0]
        return results[1:]

    def close(self):
        return self.stateless_env.close()
# TODO: what about other methods
