import torch.profiler

def setup_profiler(profile_config, save_path):
    """Instantiate a torch profiler according to CLI/config settings.

    Args:
        profile_config: Hydra config node with ``profile_*`` attributes.
        save_path: Base experiment directory for saving traces.

    Returns:
        Active ``torch.profiler.profile`` context or ``None`` if disabled.
    """

    if profile_config.profile:
        print(f"Profiling enabled. Will start after {profile_config.profile_wait} steps, with {profile_config.profile_active} active steps.")
        print(f"Will be saved to {save_path}/profiles/")

        profiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=profile_config.profile_wait,
                warmup=3,
                active=profile_config.profile_active,
                repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                f"{save_path}/profiles/"
            ),
            record_shapes=True,
            with_stack=True,
        )
        profiler.start()
        return profiler
    else:
        return None
