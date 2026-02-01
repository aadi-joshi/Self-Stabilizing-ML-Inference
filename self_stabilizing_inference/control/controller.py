class Controller:
    def __init__(self, cooldown_steps):
        self.cooldown = 0
        self.cooldown_steps = cooldown_steps

    def decide(self, degraded):
        if self.cooldown > 0:
            self.cooldown -= 1
            return None

        if degraded:
            self.cooldown = self.cooldown_steps
            return "SWITCH_TO_ROBUST"

        return None
class Controller:
    def __init__(self, cooldown_steps):
        self.cooldown = 0
        self.cooldown_steps = cooldown_steps

    def decide(self, degraded):
        if self.cooldown > 0:
            self.cooldown -= 1
            return None

        if degraded:
            self.cooldown = self.cooldown_steps
            return "SWITCH_TO_ROBUST"

        return None
