import threading
from typing import Optional, Any
from .config.settings import AppConfig
from .unitree_utils import initialize_unitree_channel

class AgentActions:
    def __init__(self, config: AppConfig):
        self._config = config
        self._arm_client: Optional[Any] = None
        self._lock = threading.Lock()

    def _ensure_arm_client(self) -> bool:
        if self._arm_client is not None:
            return True

        if not initialize_unitree_channel(self._config):
            return False

        try:
            from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient
            
            client = G1ArmActionClient()
            if hasattr(client, "SetTimeout"):
                client.SetTimeout(10.0)
            client.Init()
            self._arm_client = client
            return True
        except Exception as e:
            print(f"Failed to initialize Unitree G1 arm action client: {e}")
            self._arm_client = None
            return False

    def trigger_shake_hand(self) -> None:
        """Asynchronously trigger the shake hand action on the robot."""
        def _run_action():
            with self._lock:
                if not self._ensure_arm_client():
                    return
                # Action ID 27 corresponds to "shake hand"
                try:
                    ret = self._arm_client.ExecuteAction(27)
                    if ret != 0:
                        print(f"Failed to execute shake hand action, error code: {ret}")
                except Exception as e:
                    print(f"Error triggering shake hand action: {e}")

        # Run in a background thread so it doesn't block audio/voice streaming
        t = threading.Thread(target=_run_action, daemon=True)
        t.start()
