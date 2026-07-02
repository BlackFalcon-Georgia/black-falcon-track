# BLACK FALCON — Visual Object Detection & Tracking

Phase 1 prototype: general-purpose object detection with tap-to-track
selection, built for two use cases:

- 🛡️ **თავდაცვითი** — საფრენი ობიექტების (დრონების) დეტექცია და თრექინგი
- 🚔 **სამართალდამცავი** — მანქანის/ობიექტის მონიშვნა და მიდევნება

## სტრუქტურა

```
black-falcon-track/
├── backend/     FastAPI + YOLOv8n object detection API (deploy → Render.com)
├── web/         ერთი-ფაილიანი ვებ დემო, კამერით ტესტირებისთვის ბრაუზერში
└── mobile/      Expo (React Native) აპლიკაციის სქაფოლდი
```

## როგორ გავუშვათ

### 1. Backend (Render.com)

1. ატვირთე `backend/` ფოლდერი ცალკე Render.com web service-ად
   (ან დაუკავშირე ეს repo პირდაპირ Render-ს)
2. Build command: `pip install -r requirements.txt`
3. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. პირველი გაშვება ცოტა ხანს გასტანს — მოდელი (`yolov8n.pt`) ავტომატურად
   ჩამოიტვირთება პირველ მოთხოვნაზე
5. მიიღებ URL-ს, მაგ: `https://black-falcon-backend.onrender.com`

### 2. Web დემო

1. გახსენი `web/index.html` ბრაუზერში (უბრალოდ double-click, ან
   GitHub Pages-ზეც შეიძლება host გაუკეთო)
2. ჩაწერე backend-ის URL ველში
3. დააჭირე "📷 კამერის ჩართვა"
4. დააჭირე ეკრანზე ნებისმიერ აღმოჩენილ ობიექტს (🟩 მწვანე ჩარჩო) — გახდება
   🟥 მონიშნული და თრექირებადი

### 3. Mobile (Expo)

```bash
cd mobile
npm install
npx expo start
```

გახსენი Expo Go აპით ტელეფონზე, ან `eas build`-ით ააწყვე APK
(შენი დანარჩენი პროექტების მსგავსად).

## რა მუშაობს ახლა (Phase 1)

- ✅ ზოგადი ობიექტების დეტექცია (COCO კლასები — ადამიანი, მანქანა,
  თვითმფრინავი, ჩიტი და ა.შ., pretrained YOLOv8n მოდელით)
- ✅ ეკრანზე შეხებით ობიექტის მონიშვნა
- ✅ მარტივი frame-to-frame თრექინგი (უახლოესი ცენტროიდის დამთხვევით)

## რა არის შემდეგი ეტაპი (Phase 2+)

- 🔲 დრონისთვის სპეციალურად გაწვრთნილი მოდელი (custom dataset საჭიროა —
  pretrained მოდელს არა აქვს "drone" კლასი ჩაშენებული)
- 🔲 რეალური Re-identification (ობიექტის ცნობა კადრიდან გაქრობის შემდეგ)
- 🔲 მანქანის ნომრის ამოცნობა (ANPR/LPR მოდული)
- 🔲 რუკაზე მარშრუტის ვიზუალიზაცია

## შენიშვნა

ეს არის **დეტექცია/თრექინგის** სისტემა — არა სიგნალის ჩახშობის ან
საწინააღმდეგო (jamming/spoofing) სისტემა. ეს უკანასკნელი მოითხოვს
hardware ინჟინერიას და სახელმწიფო ავტორიზაციას და არ შედის ამ repo-ს
scope-ში.
