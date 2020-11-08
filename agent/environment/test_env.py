from habitatenv import HabitatEnv

env = HabitatEnv("bench/Roane.glb")
env.goto_state([0.5, 0, -0.5, 1, 0, 0, 0])
