from agent.environment.environment import Environment
#from agent.environment.ai2thor import THORDiscreteEnvironment
from agent.environment.habitat import HabitatDiscreteEnvironment


class AI2ThorEnvironment:
    '''
    name:
        Can be one of those:
            # Arkansaw
            # Mesic
    '''
    def __init__(self, name = "Arkansaw", grid_size = 0.25):
        import ai2thor.controller
        self.name = name
        self.grid_size = grid_size
        self.controller = ai2thor.controller.Controller()
        self.state = None       

    def start(self):
        self.controller.start()
        self.reset()

    def reset(self):
        self.controller.reset(self.name)

        # gridSize specifies the coarseness of the grid that the agent navigates on
        self.state = self.controller.step(dict(action='Initialize', gridSize=self.grid_size))

    def step(self):
        self.state = self.controller.step(dict(action='MoveAhead'))

    def render(self, mode = "rgb_array"):
        if mode == "rgb_array":
            return self.state.frame

    def render_target(self, mode = "rgb_array"):
        if mode == "rgb_array":
            return self.state

def make(name):
    if name == "unity":
        return AI2ThorEnvironment()