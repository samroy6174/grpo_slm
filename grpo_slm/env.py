import gym
import random

class ReasoningEnv(gym.Env):
    """
    A tiny arithmetic reasoning environment. At each step, it returns
    a simple math prompt (e.g. '2+3') and the agent must return the correct
    integer. Reward = 1 if correct, else 0.
    """
    def __init__(self, max_value=10):
        super().__init__()
        self.max = max_value
        self.observation_space = gym.spaces.Discrete(self.max**2)
        self.action_space = gym.spaces.Discrete(self.max * 2)
        self.current = None

    def reset(self):
        a = random.randint(1, self.max)
        b = random.randint(1, self.max)
        op = random.choice(['+', '-'])
        self.current = f"{a}{op}{b}"
        return self.current

    def step(self, action):
        # action is integer; we map 0..max*2-1 to possible answers
        prompt = self.current
        true = eval(prompt)
        pred = action
        reward = 1.0 if pred == true else 0.0
        done = True
        info = {'true': true}
        return None, reward, done, info
