"""
BLACK FALCON — ANPR/LPR Module (Automatic Number Plate Recognition)
======================================================================
დანიშნულება: მანქანის (car) ჩარჩოს შიგნით ვპოულობთ ნომრის ბლოკს და
ვცნობთ მასზე დაბეჭდილ ტექსტს.

ორეტაპიანი მიდგომა (მეხსიერების დაზოგვისთვის):

  1. ლოკალიზაცია (classical CV, არა AI-model) —
     grayscale → edge detection → contour search → aspect-ratio ფილტრი
     (ნომრის ბლოკი ჩვეულებრივ 2:1-დან 6:1-მდე თანაფარდობისაა)
     ეს საფეხური არ საჭიროებს არანაირ მოდელს, თითქმის უფასოა RAM-ისთვის.

  2. OCR (EasyOCR) — ტექსტის ამოცნობა, **მხოლოდ** მცირე, უკვე
     ლოკალიზებულ მონაკვეთზე (არა მთელ სურათზე) — ეს მნიშვნელოვნად
     ამცირებს გამოთვლით/მეხსიერების დატვირთვას, ვიდრე EasyOCR-ის
     მთელ კადრზე გაშვება.

⚠️ მეხსიერების გაფრთხილება: EasyOCR თავისთავად ~300-400MB RAM-ს
საჭიროებს პირველივე ჩატვირთვაზე. Free tier-ის 512MB ლიმიტთან ერთად
ეს **სერიოზული რისკია** — რეკომენდებულია ტესტირება ხდებოდეს
ცალკე, არა ერთდროულად /detect ან /detect_drone-თან.
"""

from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from pydantic import BaseModel


class PlateResult(BaseModel):
    car_x1: float
    car_y1: float
    car_x2: float
    car_y2: float
    plate_x1: Optional[float] = None
    plate_y1: Optional[float] = None
    plate_x2: Optional[float] = None
    plate_y2: Optional[float] = None
    plate_text: Optional[str] = None
    plate_confidence: Optional[float] = None
    status: str  # "found" | "no_plate_localized" | "ocr_failed"


# ----------------------------------------------------------------------
# ლაზი-ჩატვირთვადი OCR reader (singleton, ისევე როგორც YOLO მოდელები)
# ----------------------------------------------------------------------
_ocr_reader = None


def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        # english alphanumeric საკმარისია უმეტესი ნომრის ფორმატისთვის
        # (ლათინური ასოები + ციფრები); gpu=False — Render-ის CPU-only გარემოსთვის
        _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _ocr_reader


# ----------------------------------------------------------------------
# ეტაპი 1: პლატის ლოკალიზაცია (classical CV)
# ----------------------------------------------------------------------
def localize_plate(car_crop: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """
    ვცდილობთ ნომრის ბლოკის მართკუთხედის პოვნას მანქანის crop-ში.
    აბრუნებს (x1, y1, x2, y2) car_crop-ის საკუთარ კოორდინატებში,
    ან None თუ ვერაფერი მოვძებნეთ.
    """
    import cv2

    arr = np.array(car_crop.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(gray, 30, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    h, w = gray.shape
    best_box = None
    best_score = 0.0

    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if ch == 0:
            continue
        aspect = cw / ch

        # ნომრის ბლოკის ტიპური თანაფარდობა და ზომა (car crop-ის შიგნით,
        # ჩვეულებრივ მანქანის ქვედა 60%-ში მდებარეობს)
        plausible_aspect = 2.0 <= aspect <= 6.0
        plausible_size = (0.15 * w) <= cw <= (0.9 * w) and (0.03 * h) <= ch <= (0.35 * h)
        plausible_position = y >= h * 0.35  # ნომერი იშვიათად არის მანქანის ზედა ნახევარში

        if plausible_aspect and plausible_size and plausible_position:
            # score-ით ვირჩევთ ყველაზე "პლატისმაგვარ" კანდიდატს
            score = cw * ch
            if score > best_score:
                best_score = score
                best_box = (x, y, x + cw, y + ch)

    return best_box


# ----------------------------------------------------------------------
# ეტაპი 2: OCR — ტექსტის ამოცნობა ლოკალიზებულ მონაკვეთზე
# ----------------------------------------------------------------------
def run_ocr(plate_crop: Image.Image) -> Tuple[Optional[str], float]:
    reader = get_ocr_reader()
    arr = np.array(plate_crop.convert("RGB"))
    results = reader.readtext(arr, detail=1, paragraph=False)

    if not results:
        return None, 0.0

    # ავირჩიოთ ყველაზე მაღალი confidence-ის ტექსტი (ჩვეულებრივ ერთი
    # მთლიანი სტრიქონია ნომერზე)
    best = max(results, key=lambda r: r[2])
    text = best[1].strip().upper().replace(" ", "")
    confidence = float(best[2])
    return text, confidence


# ----------------------------------------------------------------------
# მთავარი ფუნქცია — car detection-იდან სრულ pipeline-მდე
# ----------------------------------------------------------------------
def recognize_plate(image: Image.Image, car_box: Tuple[float, float, float, float]) -> PlateResult:
    x1, y1, x2, y2 = car_box
    ix1, iy1 = max(0, int(x1)), max(0, int(y1))
    ix2, iy2 = min(image.width, int(x2)), min(image.height, int(y2))

    result = PlateResult(car_x1=x1, car_y1=y1, car_x2=x2, car_y2=y2, status="no_plate_localized")

    if ix2 <= ix1 or iy2 <= iy1:
        return result

    car_crop = image.crop((ix1, iy1, ix2, iy2))
    plate_box = localize_plate(car_crop)

    if plate_box is None:
        return result

    px1, py1, px2, py2 = plate_box
    plate_crop = car_crop.crop((px1, py1, px2, py2))

    # OCR-ისთვის სასარგებლოა ცოტა გადიდება, თუ პლატის crop პატარაა
    if plate_crop.width < 200:
        scale = 200 / max(1, plate_crop.width)
        plate_crop = plate_crop.resize(
            (int(plate_crop.width * scale), int(plate_crop.height * scale))
        )

    try:
        text, confidence = run_ocr(plate_crop)
    except Exception:
        result.status = "ocr_failed"
        return result

    if text:
        result.plate_x1 = ix1 + px1
        result.plate_y1 = iy1 + py1
        result.plate_x2 = ix1 + px2
        result.plate_y2 = iy1 + py2
        result.plate_text = text
        result.plate_confidence = confidence
        result.status = "found"

    return result


def recognize_plates_in_image(image: Image.Image, car_boxes: List[Tuple[float, float, float, float]]) -> List[PlateResult]:
    """ყველა 'car' ჩარჩოსთვის ვცდილობთ ნომრის ამოცნობას."""
    return [recognize_plate(image, box) for box in car_boxes]
