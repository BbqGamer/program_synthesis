from gymnasium.envs.registration import register

register(
    id="Min2Game",
    entry_point="assembly_game.environments:Min2Game",
    disable_env_checker=True,
    order_enforce=False,
)

register(
    id="Min3Game",
    entry_point="assembly_game.environments:Min3Game",
    disable_env_checker=True,
    order_enforce=False,
)
