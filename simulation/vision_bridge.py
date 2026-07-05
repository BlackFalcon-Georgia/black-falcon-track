"""
BLACK FALCON — Vision-to-Mission Bridge
=========================================
ეს ფაილი აკავშირებს ორ ნაწილს:

  1. რეალურ YOLO დეტექციას (backend-ის /detect ან /detect_drone) —
     რასაც კამერა "ხედავს" (bounding box, ეკრანზე)

  2. mission_simulator.py-ის state machine-ს — რომელსაც სჭირდება
     GPS-ის მსგავსი "სამიზნის პოზიცია", არა უბრალო ეკრანის კოორდინატები

ეს კვლავ **სიმულაციური/თეორიული დონეა** — დრონის საკუთარი GPS მდებარეობა
და heading (მიმართულება) აქაც სიმულირებულია, მაგრამ ვიზუალური დეტექცია
**რეალურია** (რეალურ backend-ს ვეკითხებით). რეალურ hardware-ზე მხოლოდ
`drone_gps`/`drone_heading` ცვლადები შეიცვლება Pixhawk-იდან წამოსული
ნამდვილი ტელემეტრიით (MAVLink-ით).

გამოთვლის პრინციპი (monocular distance estimation):
  - ცნობილია სამიზნის საშუალო ზომა რეალურ ცხოვრებაში (მაგ. დრონი ~0.5მ)
  - ცნობილია კამერის ხედვის კუთხე (FOV)
  - bounding box-ის ზომა ეკრანზე + ეს ორი მუდმივა → მიახლოებითი მანძილი
  - bounding box-ის ჰორიზონტალური პოზიცია ეკრანზე → მიახლოებითი bearing
    (რამდენით არის სამიზნე მარცხნივ/მარჯვნივ დრონის "წინ" მიმართულებიდან)

ეს არის სტანდარტული ტექნიკა (monocular pinhole camera distance estimation),
რომელსაც იყენებენ დაბალბიუჯეტიან drone/robotics პროექტებში, სადაც
stereo/depth კამერა არ არის ხელმისაწვდომი.
"""

import math
import io
from dataclasses import dataclass
from typing import Optional, List

from mission_simulator import (
    GPSPoint,
    DroneMissionSimulator,
    MissionState,
    SIM_TICK_S,
)


# ----------------------------------------------------------------------
# კამერისა და სამიზნის კონსტანტები — ეს დაკალიბრდება რეალურ hardware-ზე
# ----------------------------------------------------------------------
CAMERA_HFOV_DEG = 60.0          # კამერის ჰორიზონტალური ხედვის კუთხე
KNOWN_OBJECT_WIDTH_M = {        # საშუალო ზომა რეალურ ცხოვრებაში
    "drone": 0.45,
    "car": 1.8,
    "person": 0.5,
}
DEFAULT_OBJECT_WIDTH_M = 0.5


@dataclass
class Detection:
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


def estimate_distance_m(box_width_px: float, image_width_px: float,
                         class_name: str = "drone") -> float:
    """
    Monocular distance estimation — უბრალო, კამერის ხედვის კუთხეზე
    დაფუძნებული გამოთვლა (არ არის stereo-ის დონის ზუსტი, მაგრამ
    გამოსადეგია მიახლოებითი მანძილისთვის).
    """
    if box_width_px <= 0:
        return float("inf")

    real_width_m = KNOWN_OBJECT_WIDTH_M.get(class_name, DEFAULT_OBJECT_WIDTH_M)

    # პიქსელი-რადიან კონვერტაცია კამერის FOV-დან
    hfov_rad = math.radians(CAMERA_HFOV_DEG)
    # focal length პიქსელებში (pinhole camera model)
    focal_px = (image_width_px / 2) / math.tan(hfov_rad / 2)

    distance_m = (real_width_m * focal_px) / box_width_px
    return distance_m


def estimate_bearing_deg(box_center_x_px: float, image_width_px: float) -> float:
    """
    ბრუნვის კუთხე დრონის "წინ" მიმართულებიდან — უარყოფითი მარცხნივ,
    დადებითი მარჯვნივ.
    """
    normalized = (box_center_x_px - image_width_px / 2) / (image_width_px / 2)
    return normalized * (CAMERA_HFOV_DEG / 2)


def detection_to_gps(detection: Detection, image_width_px: float,
                     drone_pos: GPSPoint, drone_heading_deg: float) -> GPSPoint:
    """
    ითვლის სამიზნის მიახლოებით GPS პოზიციას drone-ის ცნობილი
    (თუნდაც სიმულირებული) პოზიციისა და მიმართულებისგან.
    """
    box_width_px = detection.x2 - detection.x1
    box_center_x = (detection.x1 + detection.x2) / 2

    distance_m = estimate_distance_m(box_width_px, image_width_px, detection.class_name)
    relative_bearing = estimate_bearing_deg(box_center_x, image_width_px)
    absolute_bearing_deg = drone_heading_deg + relative_bearing

    dx = distance_m * math.sin(math.radians(absolute_bearing_deg))
    dy = distance_m * math.cos(math.radians(absolute_bearing_deg))

    d_lat = dy / 111_111.0
    d_lon = dx / (111_111.0 * math.cos(math.radians(drone_pos.lat)))
    return GPSPoint(drone_pos.lat + d_lat, drone_pos.lon + d_lon)


def pick_best_detection(detections: List[Detection], target_class: str = "drone") -> Optional[Detection]:
    """ვირჩევთ ყველაზე მაღალი confidence-ის დეტექციას სწორი კლასიდან."""
    candidates = [d for d in detections if d.class_name == target_class]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.confidence)


# ----------------------------------------------------------------------
# Backend-თან რეალური კომუნიკაცია
# ----------------------------------------------------------------------
def call_backend_detect(image_bytes: bytes, api_url: str, endpoint: str = "detect_drone") -> List[Detection]:
    """
    გზავნის სურათს ჩვენს რეალურ, დეპლოირებულ backend-ს
    (black-falcon-track.onrender.com) და აბრუნებს Detection ობიექტების სიას.
    """
    import requests
    url = f"{api_url.rstrip('/')}/{endpoint}"
    files = {"file": ("frame.jpg", image_bytes, "image/jpeg")}
    resp = requests.post(url, files=files, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return [
        Detection(
            class_name=d["class_name"],
            confidence=d["confidence"],
            x1=d["x1"], y1=d["y1"], x2=d["x2"], y2=d["y2"],
        )
        for d in data.get("detections", [])
    ]


# ----------------------------------------------------------------------
# მთავარი loop — ერთ frame-ს იღებს და ერთ mission step-ს აკეთებს
# ----------------------------------------------------------------------
def run_vision_guided_step(
    sim: DroneMissionSimulator,
    detections: List[Detection],
    image_width_px: float,
    drone_heading_deg: float,
    target_class: str = "drone",
) -> Optional[GPSPoint]:
    """
    ერთი "ნაბიჯი" — რეალური დეტექციებიდან ვირჩევთ სამიზნეს, ვითვლით
    მის მიახლოებით GPS პოზიციას, და ვაწვდით mission_simulator-ს.

    აბრუნებს სამიზნის გამოთვლილ GPS პოზიციას (ან None, თუ არაფერი
    ჩანდა ამ frame-ზე).
    """
    best = pick_best_detection(detections, target_class)

    if best is None:
        sim.step(target_pos=None, target_visible=False)
        return None

    target_gps = detection_to_gps(best, image_width_px, sim.drone_pos, drone_heading_deg)
    sim.step(target_pos=target_gps, target_visible=True)
    return target_gps


# ----------------------------------------------------------------------
# ტესტი — სინთეზური დეტექციებით (camera/hardware გარეშე), რომ
# დავამტკიცოთ მანძილის/bearing-ის გამოთვლა მუშაობს სწორად
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("ტესტი 1: მანძილის გამოთვლის სისწორე")
    print("=" * 60)

    IMAGE_WIDTH = 1280.0

    # სცენარი: ცნობილი drone სიგანე 0.45მ, გამოვთვალოთ სხვადასხვა
    # box_width_px-ზე რა მანძილი გამოვა — ლოგიკურად, პატარა box = შორს
    for box_width_px in [200, 100, 50, 25, 10]:
        dist = estimate_distance_m(box_width_px, IMAGE_WIDTH, "drone")
        print(f"box_width={box_width_px:>4}px  →  estimated distance = {dist:6.1f}მ")

    print()
    print("=" * 60)
    print("ტესტი 2: სრული ინტეგრაცია — 'დრონი მოახლოვდება' სცენარი")
    print("=" * 60)

    launch = GPSPoint(lat=41.7151, lon=44.8271)
    sim = DroneMissionSimulator(launch)

    # ვასიმულირებთ, რომ box_width თანდათან იზრდება (სამიზნე ახლოვდება)
    # და მერე შორდება (გადადის 2კმ-ის იქითაც)
    box_widths = list(range(15, 60, 3)) + list(range(60, 5, -3))
    for i, bw in enumerate(box_widths):
        fake_detection = Detection(
            class_name="drone",
            confidence=0.9,
            x1=IMAGE_WIDTH / 2 - bw / 2,
            x2=IMAGE_WIDTH / 2 + bw / 2,
            y1=300, y2=300 + bw,
        )
        target_gps = run_vision_guided_step(
            sim, [fake_detection], IMAGE_WIDTH, drone_heading_deg=0.0
        )
        dist_from_launch = sim.drone_pos.distance_to(launch)
        print(f"[t={i:>2}] box_w={bw:>3}px  state={sim.state.value:<16} "
              f"launch-დან: {dist_from_launch:6.1f}მ")

    print()
    print(f"საბოლოო state: {sim.state.value}")
    print(f"ლოგის ჩანაწერები: {len(sim.log.entries)}")
