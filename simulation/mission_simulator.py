"""
BLACK FALCON — Mission Logic Simulator
========================================
ეს არის Pixhawk/Jetson-ის HARDWARE-ის გარეშე სიმულაცია — ის ამოწმებს
თავად ლოგიკას (state machine), სანამ ეს კოდი გადავიდოდა რეალურ
drone-ზე.

რას სიმულირებს:
  1. დრონი "მისდევს" მოძავე სამიზნეს (მაგ. მანქანას)
  2. ინარჩუნებს სამიზნო დისტანციას (follow_distance_m)
  3. თვალყურს ადევნებს launch point-იდან მანძილს
  4. თუ გადააჭარბა max_range_km-ს (ან სამიზნე დაიკარგა) —
     ინახავს GPS ლოგს და გადადის "დაბრუნების" რეჟიმში
  5. აჩვენებს მთელ მისიას ტექსტური ლოგის სახით + ვიზუალურ გრაფიკს

გაშვება:
    python mission_simulator.py
"""

import math
import time
import random
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ----------------------------------------------------------------------
# კონფიგურაცია — ეს რიცხვები მოგვიანებით რეალურ პარამეტრებად გახდება
# ----------------------------------------------------------------------
FOLLOW_DISTANCE_M = 15.0      # რა მანძილზე უნდა "მისდევდეს" სამიზნეს
MAX_RANGE_KM = 2.0            # launch point-იდან მაქსიმალური დისტანცია
TARGET_LOST_TIMEOUT_S = 5.0   # რამდენ წამში ჩაითვალოს "დაიკარგა"
SIM_TICK_S = 1.0              # სიმულაციის ერთი "ნაბიჯის" ხანგრძლივობა
DRONE_MAX_SPEED_MPS = 12.0    # დრონის მაქსიმალური სისწრაფე (მ/წმ)


class MissionState(Enum):
    IDLE = "IDLE"
    FOLLOWING = "FOLLOWING"
    SEARCHING = "SEARCHING"
    RETURNING_HOME = "RETURNING_HOME"
    LANDED = "LANDED"


@dataclass
class GPSPoint:
    lat: float
    lon: float

    def distance_to(self, other: "GPSPoint") -> float:
        """დაახლოებითი მანძილი მეტრებში (haversine ფორმულა)."""
        R = 6371000  # დედამიწის რადიუსი მეტრებში
        lat1, lon1 = math.radians(self.lat), math.radians(self.lon)
        lat2, lon2 = math.radians(other.lat), math.radians(other.lon)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return 2 * R * math.asin(math.sqrt(a))


@dataclass
class MissionLog:
    entries: list = field(default_factory=list)

    def add(self, tick: int, state: MissionState, message: str, drone: GPSPoint, target: Optional[GPSPoint] = None):
        self.entries.append({
            "tick": tick,
            "state": state.value,
            "message": message,
            "drone_gps": {"lat": drone.lat, "lon": drone.lon},
            "target_gps": {"lat": target.lat, "lon": target.lon} if target else None,
        })

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.entries, f, indent=2, ensure_ascii=False)

    def print_summary(self):
        for e in self.entries:
            print(f"[t={e['tick']:>3}] {e['state']:<16} {e['message']}")


class DroneMissionSimulator:
    """
    Ეს კლასი "თამაშობს" ორივე მხარეს — თავად დრონსაც და მოძრავ
    სამიზნესაც — უბრალო წრფივი გადაადგილების მოდელით, რომ ვცადოთ
    state machine-ის ლოგიკა.
    """

    def __init__(self, launch: GPSPoint):
        self.launch = launch
        self.drone_pos = GPSPoint(launch.lat, launch.lon)
        self.state = MissionState.IDLE
        self.log = MissionLog()
        self.time_since_target_seen = 0.0
        self.last_known_target: Optional[GPSPoint] = None
        self.tick = 0

    def meters_to_gps_delta(self, dx_m: float, dy_m: float, origin: GPSPoint):
        """მარტივი, მცირე მანძილებისთვის საკმარისად ზუსტი გადაქცევა."""
        d_lat = dy_m / 111_111.0
        d_lon = dx_m / (111_111.0 * math.cos(math.radians(origin.lat)))
        return GPSPoint(origin.lat + d_lat, origin.lon + d_lon)

    def step(self, target_pos: Optional[GPSPoint], target_visible: bool):
        self.tick += 1

        if self.state == MissionState.IDLE:
            if target_visible:
                self.state = MissionState.FOLLOWING
                self.log.add(self.tick, self.state, "სამიზნე მოინიშნა — მიდევნება დაიწყო", self.drone_pos, target_pos)
            return

        if self.state == MissionState.FOLLOWING:
            if not target_visible:
                self.time_since_target_seen += SIM_TICK_S
                if self.time_since_target_seen >= TARGET_LOST_TIMEOUT_S:
                    self.log.add(self.tick, MissionState.SEARCHING,
                                 f"სამიზნე არ ჩანს {TARGET_LOST_TIMEOUT_S}წმ+ — ვეძებ",
                                 self.drone_pos, self.last_known_target)
                    self.state = MissionState.SEARCHING
                return

            self.time_since_target_seen = 0.0
            self.last_known_target = target_pos

            # გადავამოწმოთ launch point-იდან მანძილი
            dist_from_launch_km = self.drone_pos.distance_to(self.launch) / 1000.0
            if dist_from_launch_km >= MAX_RANGE_KM:
                self.log.add(self.tick, self.state,
                             f"⚠️ მიღწეულია {MAX_RANGE_KM}კმ ლიმიტი (რეალურად: {dist_from_launch_km:.2f}კმ) — ვჩერდები",
                             self.drone_pos, target_pos)
                self._trigger_return_home("distance_limit_exceeded")
                return

            # მივიდევნოთ სამიზნეს follow_distance-ის დაცვით
            dist_to_target = self.drone_pos.distance_to(target_pos)
            if dist_to_target > FOLLOW_DISTANCE_M:
                self._move_towards(target_pos)
                self.log.add(self.tick, self.state,
                             f"მივდევ სამიზნეს (მანძილი: {dist_to_target:.1f}მ, launch-იდან: {dist_from_launch_km*1000:.0f}მ)",
                             self.drone_pos, target_pos)
            else:
                self.log.add(self.tick, self.state,
                             f"სამიზნო დისტანციაზე ვარ ({dist_to_target:.1f}მ) — ვარჩენ პოზიციას",
                             self.drone_pos, target_pos)
            return

        if self.state == MissionState.SEARCHING:
            if target_visible:
                self.state = MissionState.FOLLOWING
                self.time_since_target_seen = 0.0
                self.log.add(self.tick, self.state, "სამიზნე ხელახლა ვცანით — ვაგრძელებ მიდევნებას",
                             self.drone_pos, target_pos)
            else:
                self.time_since_target_seen += SIM_TICK_S
                if self.time_since_target_seen >= TARGET_LOST_TIMEOUT_S * 2:
                    self.log.add(self.tick, self.state,
                                 "სამიზნე საბოლოოდ დაიკარგა — ვბრუნდები",
                                 self.drone_pos, self.last_known_target)
                    self._trigger_return_home("target_lost")
            return

        if self.state == MissionState.RETURNING_HOME:
            dist_to_home = self.drone_pos.distance_to(self.launch)
            if dist_to_home < 2.0:
                self.state = MissionState.LANDED
                self.log.add(self.tick, self.state, "დრონი დაბრუნდა launch point-ზე — დაჯავშნა", self.drone_pos)
            else:
                self._move_towards(self.launch)
                self.log.add(self.tick, self.state,
                             f"ვბრუნდები launch point-ზე (დაშორება: {dist_to_home:.0f}მ)",
                             self.drone_pos)
            return

    def _trigger_return_home(self, reason: str):
        if self.last_known_target:
            print(f"📍 GPS ლოგი შენახულია: lat={self.last_known_target.lat:.6f}, "
                  f"lon={self.last_known_target.lon:.6f} (მიზეზი: {reason})")
        self.state = MissionState.RETURNING_HOME

    def _move_towards(self, dest: GPSPoint):
        dist = self.drone_pos.distance_to(dest)
        step_dist = min(DRONE_MAX_SPEED_MPS * SIM_TICK_S, dist)
        if dist == 0:
            return
        frac = step_dist / dist
        new_lat = self.drone_pos.lat + (dest.lat - self.drone_pos.lat) * frac
        new_lon = self.drone_pos.lon + (dest.lon - self.drone_pos.lon) * frac
        self.drone_pos = GPSPoint(new_lat, new_lon)


def simulate_moving_target(launch: GPSPoint, sim: DroneMissionSimulator, ticks: int = 400):
    """
    სამიზნეს (მაგ. მანქანის) მარტივი მოძრაობის მოდელი — შორდება
    launch point-ს მუდმივი მიმართულებით, ვიდრე არ გავცდით ლიმიტს.
    """
    target_pos = GPSPoint(launch.lat, launch.lon)
    heading_deg = random.uniform(0, 360)
    speed_mps = 8.0  # ~29 კმ/სთ, ჩვეულებრივი საქალაქო მოძრაობა

    for _ in range(ticks):
        if sim.state == MissionState.LANDED:
            break

        # სამიზნის მოძრაობა
        dx = speed_mps * math.sin(math.radians(heading_deg))
        dy = speed_mps * math.cos(math.radians(heading_deg))
        target_pos = sim.meters_to_gps_delta(dx, dy, target_pos)

        # ვიზუალური "დანახვა" — ვთვალით 95% შემთხვევაში ვხედავთ
        target_visible = random.random() < 0.95

        sim.step(target_pos, target_visible)
        time.sleep(0)  # რეალურ დროში სისუფთავისთვის — სიმულაციაში სულ 0

    return target_pos


if __name__ == "__main__":
    LAUNCH_POINT = GPSPoint(lat=41.7151, lon=44.8271)  # თბილისი, მაგალითი

    sim = DroneMissionSimulator(LAUNCH_POINT)
    final_target_pos = simulate_moving_target(LAUNCH_POINT, sim, ticks=500)

    print("\n" + "=" * 60)
    print("მისიის სრული ლოგი:")
    print("=" * 60)
    sim.log.print_summary()

    sim.log.save("mission_log.json")
    print(f"\n✅ სრული ლოგი შენახულია: mission_log.json ({len(sim.log.entries)} ჩანაწერი)")
    print(f"საბოლოო მდგომარეობა: {sim.state.value}")
