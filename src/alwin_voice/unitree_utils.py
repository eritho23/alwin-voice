import importlib
from .config.settings import AppConfig

_unitree_channel_initialized = False

def initialize_unitree_channel(config: AppConfig) -> bool:
    global _unitree_channel_initialized
    if _unitree_channel_initialized:
        return True
    
    try:
        channel_module = importlib.import_module("unitree_sdk2py.core.channel")
        init_fn = getattr(channel_module, "ChannelFactoryInitialize")
        iface = config.unitree_net_iface or ""
        if iface:
            init_fn(0, iface)
        else:
            init_fn(0)
        _unitree_channel_initialized = True
        return True
    except Exception as e:
        print(f"Failed to initialize Unitree channel: {e}")
        return False
