from gymnasium.envs.registration import register

register(
    id="MinGame",
    entry_point="assembly_game.environments:MinGame",
    disable_env_checker=True,
    order_enforce=False,
)
