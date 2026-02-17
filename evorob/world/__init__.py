from gymnasium.envs.registration import register

register(
    id="PassiveWalker-v0",
    entry_point="evorob.world.envs.passive_walker:PassiveWalker",
    max_episode_steps=1000,
)

register(
    id="AntHill-v0",
    entry_point="evorob.world.envs.ant_hill_sol:AntHillEnv",
    max_episode_steps=1000,
)

__version__ = "0.1"