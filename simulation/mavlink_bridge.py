"""
BLACK FALCON — MAVLink Bridge
================================
ეს ფაილი აკავშირებს mission_simulator.py-ის გადაწყვეტილებებს
("მივდევ", "ვბრუნდები" და ა.შ.) **რეალურ MAVLink ბრძანებებთან** —
იმ პროტოკოლთან, რომელსაც Pixhawk (ან PX4-ის სიმულატორი) ესმის.

ორი რეჟიმი:

  DRY_RUN = True  (default, hardware/SITL გარეშე)
      ბრძანებები არ იგზავნება რეალურად — უბრალოდ ჩაბეჭდავს, თუ რა
      გაეგზავნებოდა. ეს საშუალებას გვაძლევს დავამტკიცოთ ლოგიკა
      მთლიანად უფასოდ, drone/simulator-ის გარეშეც.

  DRY_RUN = False (საჭიროებს გაშვებულ PX4 SITL-ს ან რეალურ Pixhawk-ს)
      რეალურად უკავშირდება MAVLink connection-ს და გზავნის ბრძანებებს.

როგორ გავუშვათ რეალურ PX4 SITL-თან (უფასო, hardware-ის გარეშე):
  1. დააინსტალირე PX4 SITL:  https://docs.px4.io/main/en/dev_setup/dev_env.html
     ან უფრო მარტივად, Docker-ით:
       docker run -it --rm px4io/px4-dev-simulation-focal
  2. გაუშვი სიმულატორი:  make px4_sitl gazebo
     ეს გახსნის ვირტუალურ დრონს MAVLink პორტზე udp:14540
  3. ამ ფაილში შეცვალე: DRY_RUN = False
  4. გაუშვი:  python mavlink_bridge.py
"""

import time
from dataclasses import dataclass
from typing import Optional

from mission_simulator import GPSPoint, MissionState


DRY_RUN = True
CONNECTION_STRING = "udp:127.0.0.1:14540"  # PX4 SITL-ის სტანდარტული პორტი


@dataclass
class DroneTelemetry:
    """Pixhawk-იდან წამოსული (ან SITL-ის) რეალური ტელემეტრია."""
    gps: GPSPoint
    heading_deg: float
    altitude_m: float
    armed: bool
    mode: str


class MAVLinkBridge:
    def __init__(self, connection_string: str = CONNECTION_STRING, dry_run: bool = DRY_RUN):
        self.dry_run = dry_run
        self.connection_string = connection_string
        self.master = None
        self._log = []

        if not self.dry_run:
            self._connect()

    def _connect(self):
        from pymavlink import mavutil
        print(f"🔌 ვუკავშირდები: {self.connection_string} ...")
        self.master = mavutil.mavlink_connection(self.connection_string)
        self.master.wait_heartbeat()
        print("✅ Heartbeat მიღებულია — კავშირი დამყარებულია")

    def _record(self, action: str, detail: str):
        entry = f"[MAVLink{'·DRY-RUN' if self.dry_run else ''}] {action}: {detail}"
        self._log.append(entry)
        print(entry)

    # ------------------------------------------------------------------
    # ტელემეტრიის წაკითხვა
    # ------------------------------------------------------------------
    def get_telemetry(self) -> Optional[DroneTelemetry]:
        if self.dry_run:
            # DRY_RUN-ში ტელემეტრია გარედან მოდის (mission_simulator-ისგან)
            return None

        msg = self.master.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=2)
        if msg is None:
            return None
        heading_msg = self.master.recv_match(type="VFR_HUD", blocking=False)
        return DroneTelemetry(
            gps=GPSPoint(msg.lat / 1e7, msg.lon / 1e7),
            heading_deg=heading_msg.heading if heading_msg else 0.0,
            altitude_m=msg.relative_alt / 1000.0,
            armed=True,
            mode="UNKNOWN",
        )

    # ------------------------------------------------------------------
    # ბრძანებები — თითოეული შეესაბამება mission_simulator-ის state-ს
    # ------------------------------------------------------------------
    def arm_and_takeoff(self, altitude_m: float = 20.0):
        self._record("ARM + TAKEOFF", f"სამიზნო სიმაღლე: {altitude_m}მ")
        if self.dry_run:
            return
        from pymavlink import mavutil
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0,
        )
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, altitude_m,
        )

    def goto_position(self, target: GPSPoint, altitude_m: float = 20.0):
        """FOLLOWING state — 'მიდი ამ GPS წერტილთან'."""
        self._record("GOTO", f"lat={target.lat:.6f}, lon={target.lon:.6f}, alt={altitude_m}მ")
        if self.dry_run:
            return
        from pymavlink import mavutil
        self.master.mav.set_position_target_global_int_send(
            0, self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0b0000111111111000,  # position ONLY (ignore velocity/accel)
            int(target.lat * 1e7), int(target.lon * 1e7), altitude_m,
            0, 0, 0, 0, 0, 0, 0, 0,
        )

    def hold_position(self):
        """SEARCHING state — 'გაჩერდი, მაგრამ ჰაერზე იყავი'."""
        self._record("HOLD", "ვხტვირთავ HOLD/LOITER რეჟიმს")
        if self.dry_run:
            return
        from pymavlink import mavutil
        self.master.set_mode_apm("LOITER") if hasattr(self.master, "set_mode_apm") else None

    def return_to_launch(self):
        """RETURNING_HOME state — სტანდარტული PX4/ArduPilot 'RTL' ბრძანება."""
        self._record("RTL", "Return-To-Launch რეჟიმზე გადართვა")
        if self.dry_run:
            return
        from pymavlink import mavutil
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
            0, 0, 0, 0, 0, 0, 0, 0,
        )

    def land(self):
        self._record("LAND", "დაშვება მიმდინარეობს")
        if self.dry_run:
            return
        from pymavlink import mavutil
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_LAND,
            0, 0, 0, 0, 0, 0, 0, 0,
        )


def state_to_mavlink_action(bridge: MAVLinkBridge, state: MissionState, target_gps: Optional[GPSPoint]):
    """
    ეს არის ის 'თარჯიმანი' — mission_simulator-ის state → MAVLink ბრძანება.
    ზუსტად ეს ფუნქცია გახდება production-ის გულს, როცა რეალურ Pixhawk-ს
    დავუკავშირდებით.
    """
    if state == MissionState.FOLLOWING and target_gps:
        bridge.goto_position(target_gps)
    elif state == MissionState.SEARCHING:
        bridge.hold_position()
    elif state == MissionState.RETURNING_HOME:
        bridge.return_to_launch()
    elif state == MissionState.LANDED:
        bridge.land()


# ----------------------------------------------------------------------
# სრული ტესტი — mission_simulator + vision_bridge + mavlink_bridge
# ერთადერთ სცენარში, სრულად DRY-RUN რეჟიმში
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from vision_bridge import Detection, run_vision_guided_step
    from mission_simulator import DroneMissionSimulator

    print("=" * 70)
    print("სრული INTEGRATION ტესტი: Vision → Mission Logic → MAVLink (DRY-RUN)")
    print("=" * 70)

    launch = GPSPoint(lat=41.7151, lon=44.8271)
    sim = DroneMissionSimulator(launch)
    bridge = MAVLinkBridge(dry_run=True)

    bridge.arm_and_takeoff(altitude_m=20.0)

    IMAGE_WIDTH = 1280.0
    # ვასიმულირებთ სამიზნეს, რომელიც თანდათან შორდება 2კმ-მდე და გადაცდომამდე
    box_widths = [40] * 5 + [max(3, 40 - i) for i in range(1, 60)]

    for i, bw in enumerate(box_widths):
        detection = Detection(
            class_name="drone", confidence=0.9,
            x1=IMAGE_WIDTH / 2 - bw / 2, x2=IMAGE_WIDTH / 2 + bw / 2,
            y1=300, y2=300 + bw,
        )
        target_gps = run_vision_guided_step(sim, [detection], IMAGE_WIDTH, drone_heading_deg=0.0)
        state_to_mavlink_action(bridge, sim.state, target_gps)

        if sim.state == MissionState.LANDED:
            break

    print()
    print(f"საბოლოო state: {sim.state.value}")
    print(f"MAVLink ბრძანებები, რაც გაიგზავნა (dry-run ლოგში): {len(bridge._log)}")
