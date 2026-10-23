"""Local face enrollment and webcam recognition using YuNet + SFace.

Run: python face.py
CLI: python face.py --download-models | --check | --train
The database stores normalized SFace embeddings, not a retrained neural network.
"""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import sqlite3
import sys
import threading
import time
import unicodedata
import urllib.request
import uuid

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont, ImageTk
except ImportError as exc:
    raise SystemExit(
        f"Thieu thu vien: {exc}. Chay START.bat hoac: "
        "python -m pip install -r requirements.txt"
    ) from exc

BASE = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODEL_ID = "opencv-sface-2021dec-l2-v1"
DIMENSION = 128
MODELS = {
    "detector": (
        "face_detection_yunet_2023mar.onnx",
        "face_detection_yunet",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        232589,
    ),
    "recognizer": (
        "face_recognition_sface_2021dec.onnx",
        "face_recognition_sface",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        38696353,
    ),
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def normalized(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    size = float(np.linalg.norm(vector))
    if vector.shape != (DIMENSION,) or not np.isfinite(vector).all() or size < 1e-8:
        raise ValueError("Vector khuon mat khong hop le.")
    return vector / size


def person_name(value: str) -> str:
    value = unicodedata.normalize("NFC", value).strip()
    reserved = {"CON", "PRN", "AUX", "NUL"} | {
        f"{prefix}{n}" for prefix in ("COM", "LPT") for n in range(1, 10)
    }
    if (
        not value or len(value) > 70
        or not all(ch.isalnum() or ch in " _-" for ch in value)
        or not any(ch.isalnum() for ch in value)
        or value.upper() in reserved
        or value.casefold() in {"unknown", "unknow", "face_detected"}
    ):
        raise ValueError("Ten can co chu/so; chi them dau cach, _ hoac -. Khong dung unknown.")
    return value


def read_image(path: Path):
    # imdecode supports Windows paths containing Vietnamese characters.
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("Khong doc duoc anh.")
    return image


def ensure_models(folder: Path, log=print):
    folder.mkdir(parents=True, exist_ok=True)
    for filename, section, checksum, size in MODELS.values():
        target = folder / filename
        if target.exists():
            if target.stat().st_size == size and sha256(target) == checksum:
                continue
            raise ValueError(f"Model loi checksum: {target}. Doi ten file nay roi tai lai.")
        log(f"Dang tai model {filename} ({size / 1048576:.1f} MB)...")
        url = f"https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/{section}/{filename}"
        temporary = folder / (filename + f".{uuid.uuid4().hex}.part")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "LocalFaceApp/1.0"})
            with urllib.request.urlopen(request, timeout=45) as response, temporary.open("wb") as output:
                total = 0
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    total += len(block)
                    if total > size:
                        raise ValueError("Model tai ve lon hon kich thuoc du kien.")
                    output.write(block)
            if total != size or sha256(temporary) != checksum:
                raise ValueError("Model tai ve khong dung kich thuoc/checksum.")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        log(f"Da tai va kiem tra {filename}.")


class VectorDatabase:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS people (
                    name TEXT PRIMARY KEY, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS embeddings (
                    person TEXT NOT NULL REFERENCES people(name) ON DELETE CASCADE,
                    image_hash TEXT NOT NULL,
                    source TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    PRIMARY KEY(person, image_hash)
                );
            """)
            row = connection.execute("SELECT value FROM metadata WHERE key='model'").fetchone()
            if row is not None and row[0] != MODEL_ID:
                raise ValueError("Database dung model khac; hay tao database moi.")
            connection.execute("INSERT OR IGNORE INTO metadata VALUES ('model', ?)", (MODEL_ID,))

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(str(self.path), timeout=15)
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            with connection:
                yield connection
        finally:
            connection.close()

    def replace_person(self, name, entries):
        name = person_name(name)
        if len(entries) < 3:
            raise ValueError("Can it nhat 3 anh hop le khac nhau de luu mot nguoi.")
        # Validate all input before replacing any existing registration.
        rows = [(name, digest, str(source), normalized(vector).astype("<f4").tobytes())
                for digest, source, vector in entries]
        if len({row[1] for row in rows}) != len(rows):
            raise ValueError("Anh train bi trung lap.")
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO people VALUES (?, ?) ON CONFLICT(name) DO UPDATE SET updated_at=excluded.updated_at",
                (name, datetime.now(timezone.utc).isoformat()),
            )
            connection.execute("DELETE FROM embeddings WHERE person=?", (name,))
            connection.executemany("INSERT INTO embeddings VALUES (?, ?, ?, ?)", rows)

    def gallery(self):
        with self.connect() as connection:
            rows = connection.execute("SELECT person, vector FROM embeddings ORDER BY person").fetchall()
        grouped = {}
        for name, blob in rows:
            vector = normalized(np.frombuffer(blob, dtype="<f4"))
            grouped.setdefault(name, []).append(vector)
        return {name: np.stack(vectors) for name, vectors in grouped.items()}

    def list_people(self):
        with self.connect() as connection:
            return connection.execute(
                "SELECT p.name, COUNT(e.image_hash), p.updated_at FROM people p "
                "JOIN embeddings e ON p.name=e.person GROUP BY p.name ORDER BY p.name"
            ).fetchall()

    def remove(self, name):
        with self.connect() as connection:
            connection.execute("DELETE FROM people WHERE name=?", (name,))


def match_identity(vector, gallery, threshold=0.50, margin=0.06):
    if not gallery:
        return "unknown", 0.0
    query = normalized(vector)
    ranked = []
    for name, samples in gallery.items():
        scores = np.asarray(samples @ query)
        # Average the best three views, rather than accepting one accidental match.
        k = min(3, len(scores))
        ranked.append((float(np.sort(scores)[-k:].mean()), name))
    ranked.sort(reverse=True)
    best, name = ranked[0]
    competitor = ranked[1][0] if len(ranked) > 1 else -1.0
    if best < threshold or best - competitor < margin:
        return "unknown", best
    return name, best


class FaceEngine:
    def __init__(self, models: Path):
        self.detector = cv2.FaceDetectorYN.create(
            str(models / MODELS["detector"][0]), "", (320, 320), 0.90, 0.3, 5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(models / MODELS["recognizer"][0]), "")

    def detect(self, frame):
        height, width = frame.shape[:2]
        factor = min(1.0, 960 / max(width, height))
        small = cv2.resize(frame, (round(width * factor), round(height * factor))) if factor < 1 else frame
        self.detector.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.detector.detect(small)
        if faces is None:
            return []
        faces = faces.copy()
        faces[:, :14] /= factor
        return [face for face in faces if face[2] >= 30 and face[3] >= 30]

    def feature(self, frame, face):
        aligned = self.recognizer.alignCrop(frame, np.asarray(face, dtype=np.float32))
        return normalized(self.recognizer.feature(aligned))

    @staticmethod
    def quality(frame, face, minimum=70):
        height, width = frame.shape[:2]
        x, y, w, h = map(float, face[:4])
        if min(w, h) < minimum:
            return False, "Khuon mat qua nho; lai gan camera hon."
        if x < 0 or y < 0 or x + w > width or y + h > height:
            return False, "Khuon mat bi cat o vien anh."
        crop = frame[int(y):int(y+h), int(x):int(x+w)]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Normalize spatial scale before testing blur across differently sized faces.
        standard = cv2.resize(gray, (112, 112))
        if cv2.Laplacian(standard, cv2.CV_64F).var() < 25:
            return False, "Anh khuon mat qua mo."
        if not 35 <= float(gray.mean()) <= 225:
            return False, "Khuon mat qua toi hoac qua sang."
        return True, "OK"


def enroll_directory(engine, database, folder: Path, log=print):
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Khong tim thay thu muc train: {folder}")
    groups = sorted([p for p in folder.iterdir() if p.is_dir() and not p.is_symlink()])
    if not groups:
        raise ValueError("Chua co du lieu. Tao train/Ten_nguoi/ va dat anh vao do.")
    report = {"trained": [], "failed": [], "skipped": []}
    for group in groups:
        try:
            name = person_name(group.name)
        except ValueError as exc:
            report["failed"].append({"name": group.name, "reason": str(exc)})
            log(f"BO QUA {group.name}: {exc}")
            continue
        entries, seen = [], set()
        paths = sorted([p for p in group.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])
        log(f"Dang xu ly {name}: {len(paths)} anh...")
        for index, path in enumerate(paths, 1):
            try:
                digest = sha256(path)
                if digest in seen:
                    raise ValueError("Anh trung noi dung.")
                seen.add(digest)
                image = read_image(path)
                faces = engine.detect(image)
                if len(faces) != 1:
                    raise ValueError(f"Can dung 1 khuon mat, phat hien {len(faces)}.")
                valid, reason = engine.quality(image, faces[0], minimum=80)
                if not valid:
                    raise ValueError(reason)
                vector = engine.feature(image, faces[0])
                entries.append((digest, path.name, vector))
            except (ValueError, OSError, cv2.error) as exc:
                reason = str(exc)
                report["skipped"].append({"person": name, "file": path.name, "reason": reason})
                log(f"  Bo qua {path.name}: {reason}")
            if index % 10 == 0:
                log(f"  {name}: {index}/{len(paths)} anh.")
        if len(entries) < 3:
            reason = f"Chi co {len(entries)} anh hop le; can it nhat 3. Giu nguyen vector cu neu co."
            report["failed"].append({"name": name, "reason": reason})
            log(f"CHUA TRAIN {name}: {reason}")
            continue
        database.replace_person(name, entries)
        report["trained"].append({"name": name, "vectors": len(entries)})
        log(f"DA LUU {name}: {len(entries)} vector. Nen co 10-30 anh nhieu goc.")
    return report


def intersection_over_union(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    intersection = max(0, x2-x1) * max(0, y2-y1)
    union = a[2]*a[3] + b[2]*b[3] - intersection
    return intersection/union if union > 0 else 0.0


@dataclass
class Track:
    box: object
    last_seen: float
    history: deque = field(default_factory=lambda: deque(maxlen=3))


class StableTracker:
    def __init__(self):
        self.tracks = {}
        self.counter = 0

    def update(self, detections, now):
        # A gap invalidates the prior name; never keep a stale accepted identity.
        self.tracks = {k: v for k, v in self.tracks.items() if now-v.last_seen < .8}
        available = set(self.tracks)
        result = []
        for box, candidate, score, quality in detections:
            matches = [(intersection_over_union(box, self.tracks[k].box), k) for k in available]
            overlap, identifier = max(matches) if matches else (0, None)
            if overlap < .30:
                self.counter += 1
                identifier = self.counter
                self.tracks[identifier] = Track(np.array(box), now)
            else:
                available.remove(identifier)
            track = self.tracks[identifier]
            track.box, track.last_seen = np.array(box), now
            if candidate == "unknown" or not quality:
                track.history.clear()
            else:
                if track.history and track.history[-1] != candidate:
                    track.history.clear()
                track.history.append(candidate)
            accepted = candidate if len(track.history) == 3 else "unknown"
            result.append((box, accepted, score, quality, len(track.history)))
        for identifier in available:
            # A missed detection must not count toward three consecutive matches.
            self.tracks[identifier].history.clear()
        return result


def get_font(size):
    for path in (Path(os.environ.get("WINDIR", "C:/Windows"))/"Fonts/arial.ttf", Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def draw_recognition(frame, results):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = get_font(max(16, min(24, image.width//36)))
    for box, name, score, valid, count in results:
        x, y, w, h = map(int, box[:4])
        color = "#35d699" if name != "unknown" else ("#ffc64b" if count else "#ff6b79")
        draw.rectangle((x, y, x+w, y+h), outline=color, width=3)
        label = f"face_detected: {name} | cos={score:.2f}"
        if count and name == "unknown":
            label += f" ({count}/3)"
        if not valid:
            label = "face_detected: unknown | can anh ro hon"
        bound = draw.textbbox((0, 0), label, font=font)
        tw, th = bound[2]-bound[0]+12, bound[3]-bound[1]+12
        tx, ty = max(0, min(x, image.width-tw)), max(0, y-th)
        draw.rectangle((tx, ty, min(image.width, tx+tw), ty+th), fill="#111b2b")
        draw.text((tx+5, ty+3-bound[1]), label, font=font, fill=color)
    return image


class FaceApp:
    def __init__(self, root, data: Path, train: Path, camera: int):
        import tkinter as tk
        from tkinter import ttk, scrolledtext
        self.tk, self.ttk, self.root = tk, ttk, root
        self.data, self.models = data, data / "models"
        self.database = VectorDatabase(data / "face_vectors.sqlite3")
        self.events, self.frames = queue.Queue(), queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.camera_thread = None
        self.training = False
        self.closed = False
        self.raw_lock = threading.Lock()
        self.raw_frame, self.capture_valid = None, False
        self.raw_time = 0.0
        self.train_path = tk.StringVar(value=str(train))
        self.name = tk.StringVar()
        self.camera_index = tk.StringVar(value=str(camera))
        self.status = tk.StringVar(value="Sẵn sàng. Nạp ảnh và train trước khi nhận diện.")
        settings = {}
        try:
            settings = json.loads((data / "settings.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        if not isinstance(settings, dict):
            settings = {}
        def setting(key, default, low, high):
            try:
                value = float(settings.get(key, default))
                return value if low <= value <= high else default
            except (TypeError, ValueError):
                return default
        self.threshold = tk.DoubleVar(value=setting("threshold", .50, .2, .99))
        self.margin = tk.DoubleVar(value=setting("margin", .06, 0, .5))
        root.title("Face Recognition • Local Vector Database")
        root.geometry("1240x850");root.minsize(1030, 720);root.configure(bg="#0d1524")
        style = ttk.Style(root);style.theme_use("clam")
        style.configure("TFrame", background="#0d1524")
        style.configure("TLabel", background="#0d1524", foreground="#dbe5f4", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 21, "bold"), foreground="#ffffff")
        style.configure("TButton", font=("Segoe UI", 10), padding=(10, 8))
        style.configure("Treeview", rowheight=27, font=("Segoe UI", 10))
        main = ttk.Frame(root, padding=18);main.pack(fill="both", expand=True)
        ttk.Label(main, text="Nhận diện khuôn mặt", style="Title.TLabel").pack(anchor="w")
        ttk.Label(main, text="Ảnh nhiều góc → vector SQLite → webcam → tên hoặc unknown").pack(anchor="w", pady=(2, 14))
        body = ttk.Frame(main);body.pack(fill="both", expand=True)
        left=ttk.Frame(body);left.pack(side="left",fill="both",expand=True,padx=(0,16))
        right=ttk.Frame(body,width=355);right.pack(side="right",fill="y");right.pack_propagate(False)
        notebook=ttk.Notebook(right);notebook.pack(fill="both",expand=True)
        registration=ttk.Frame(notebook,padding=10)
        identities=ttk.Frame(notebook,padding=10)
        tuning=ttk.Frame(notebook,padding=10)
        notebook.add(registration,text="Đăng ký")
        notebook.add(identities,text="Database")
        notebook.add(tuning,text="Ngưỡng")
        self.preview=tk.Label(left,bg="#152236",fg="#91a6c1",text="CAMERA ĐANG TẮT\n\nNạp ảnh nhiều góc và tạo vector để bắt đầu",font=("Segoe UI",15))
        self.preview.pack(fill="both",expand=True)
        controls=ttk.Frame(left);controls.pack(fill="x",pady=10)
        ttk.Label(controls,text="Camera:").pack(side="left")
        ttk.Entry(controls,textvariable=self.camera_index,width=4).pack(side="left",padx=6)
        self.start_button=ttk.Button(controls,text="Mở camera",command=self.start_camera);self.start_button.pack(side="left",padx=3)
        ttk.Button(controls,text="Tắt camera",command=self.stop_camera).pack(side="left",padx=3)
        ttk.Button(controls,text="Chụp ảnh đăng ký",command=self.capture).pack(side="left",padx=3)
        right=registration
        ttk.Label(right,text="1  Đăng ký dữ liệu",font=("Segoe UI",12,"bold")).pack(anchor="w")
        ttk.Label(right,text="Tên người / tên thư mục").pack(anchor="w",pady=(10,2))
        ttk.Entry(right,textvariable=self.name).pack(fill="x")
        ttk.Button(right,text="Nạp ảnh nhiều góc…",command=self.import_images).pack(fill="x",pady=6)
        ttk.Label(right,text="Thư mục train").pack(anchor="w",pady=(8,2))
        ttk.Entry(right,textvariable=self.train_path).pack(fill="x")
        ttk.Button(right,text="Chọn thư mục train…",command=self.choose_train).pack(fill="x",pady=6)
        self.train_button=ttk.Button(right,text="2  Train / Tạo vector database",command=self.train);self.train_button.pack(fill="x",pady=(4,8))
        self.progress=ttk.Progressbar(right,mode="indeterminate");self.progress.pack(fill="x",pady=(0,10))
        ttk.Label(right,text="Nên có 10–30 ảnh mỗi người: chính diện,\nnghiêng trái/phải và nhìn lên/xuống nhẹ.\nMỗi ảnh chỉ có một khuôn mặt rõ.",wraplength=320).pack(anchor="w",pady=8)
        right=identities
        ttk.Label(right,text="Người đã có vector",font=("Segoe UI",11,"bold")).pack(anchor="w")
        self.people=ttk.Treeview(right,columns=("name","count"),show="headings",height=5)
        self.people.heading("name",text="Tên");self.people.heading("count",text="Vector")
        self.people.column("name",width=245);self.people.column("count",width=70,anchor="center")
        self.people.pack(fill="x",pady=6)
        ttk.Button(right,text="Xóa vector người đã chọn",command=self.remove_person).pack(fill="x")
        ttk.Label(right,text="Ảnh train vẫn giữ sau khi xóa vector.\nTrain lại sẽ đăng ký lại người trong ảnh.",wraplength=320).pack(anchor="w",pady=10)
        right=tuning
        ttk.Label(right,text="3  Ngưỡng nhận diện",font=("Segoe UI",12,"bold")).pack(anchor="w",pady=(14,6))
        options=ttk.Frame(right);options.pack(fill="x")
        ttk.Label(options,text="Cosine tối thiểu").grid(row=0,column=0,sticky="w")
        ttk.Spinbox(options,from_=.20,to=.99,increment=.01,textvariable=self.threshold,width=8).grid(row=0,column=1,padx=10,pady=3)
        ttk.Label(options,text="Chênh với người thứ hai").grid(row=1,column=0,sticky="w")
        ttk.Spinbox(options,from_=0,to=.5,increment=.01,textvariable=self.margin,width=8).grid(row=1,column=1,padx=10,pady=3)
        ttk.Label(right,text="Áp dụng khi mở lại camera. Điểm cosine\nkhông phải xác suất hoặc % chính xác.",wraplength=340).pack(anchor="w",pady=8)
        ttk.Label(right,text="Khung xanh: đã xác nhận tên\nKhung vàng: đang xác nhận 3 lần\nKhung đỏ: unknown / ảnh chưa rõ").pack(anchor="w",pady=5)
        self.log_view=scrolledtext.ScrolledText(main,height=6,font=("Consolas",9),state="disabled",bg="#121e30",fg="#b9cbe3",insertbackground="white")
        self.log_view.pack(fill="x",pady=(12,6))
        ttk.Label(main,textvariable=self.status,wraplength=1180).pack(anchor="w")
        root.protocol("WM_DELETE_WINDOW",self.close)
        self.refresh_people();root.after(70,self.poll)

    def log(self, message):
        self.events.put(("log", str(message)))

    def refresh_people(self):
        for item in self.people.get_children():self.people.delete(item)
        for name,count,_ in self.database.list_people():self.people.insert("","end",values=(name,count))

    def busy(self):
        if self.training:
            from tkinter import messagebox
            messagebox.showinfo("Đang xử lý","Đợi tạo vector hoàn tất rồi thực hiện thao tác này.")
            return True
        return False

    def choose_train(self):
        if self.busy():return
        from tkinter import filedialog
        folder=filedialog.askdirectory(title="Chọn thư mục chứa các thư mục tên người")
        if folder:self.train_path.set(folder)

    def import_images(self):
        if self.busy():return
        from tkinter import filedialog,messagebox
        try:name=person_name(self.name.get())
        except ValueError as exc:messagebox.showerror("Tên chưa hợp lệ",str(exc));return
        paths=filedialog.askopenfilenames(title=f"Chọn ảnh của {name}",filetypes=[("Ảnh","*.jpg *.jpeg *.png *.bmp *.webp")])
        if not paths:return
        folder=Path(self.train_path.get()).expanduser()/name
        folder.mkdir(parents=True,exist_ok=True)
        count=0
        for value in paths:
            source=Path(value)
            try:
                if source.suffix.lower() not in IMAGE_EXTENSIONS:continue
                read_image(source)
                destination=folder/(uuid.uuid4().hex+source.suffix.lower())
                shutil.copy2(source,destination);count+=1
            except (OSError,ValueError,cv2.error) as exc:self.log(f"Khong nap duoc {source.name}: {exc}")
        self.status.set(f"Đã nạp {count} ảnh cho {name}. Bấm Train để tạo vector.")
        self.log(f"Da nap {count} anh cho {name}; chua thay doi database.")

    def capture(self):
        if self.busy():return
        from tkinter import messagebox
        try:
            name=person_name(self.name.get())
            with self.raw_lock:
                if self.raw_frame is None or not self.capture_valid or time.monotonic()-self.raw_time>1.5:
                    raise ValueError("Mở camera, giữ đúng một khuôn mặt rõ và chụp lại.")
                frame=self.raw_frame.copy()
            folder=Path(self.train_path.get()).expanduser()/name;folder.mkdir(parents=True,exist_ok=True)
            target=folder/(datetime.now().strftime("%Y%m%d_%H%M%S_")+uuid.uuid4().hex[:8]+".jpg")
            ok,buffer=cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,95])
            if not ok:raise ValueError("Không mã hóa được ảnh.")
            buffer.tofile(str(target))
            self.status.set(f"Đã lưu ảnh của {name}. Đổi góc rồi chụp tiếp; sau đó bấm Train.")
        except (ValueError,OSError,cv2.error) as exc:messagebox.showerror("Chưa chụp được",str(exc))

    def train(self):
        if self.busy():return
        folder=Path(self.train_path.get()).expanduser()
        self.stop_camera()
        self.training=True;self.train_button.state(["disabled"]);self.start_button.state(["disabled"])
        self.progress.start(12);self.status.set("Đang tạo vector. Đọc tiến độ ở phần nhật ký.")
        def worker():
            try:
                ensure_models(self.models,self.log)
                report=enroll_directory(FaceEngine(self.models),self.database,folder,self.log)
                filename=self.data/("train_report_"+datetime.now().strftime("%Y%m%d_%H%M%S")+".json")
                filename.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
                self.events.put(("trained",report))
            except Exception as exc:self.events.put(("train_error",str(exc)))
        threading.Thread(target=worker,daemon=True,name="enroll").start()

    def remove_person(self):
        if self.busy():return
        from tkinter import messagebox
        selected=self.people.selection()
        if not selected:return
        name=self.people.item(selected[0],"values")[0]
        if messagebox.askyesno("Xóa vector",f"Xóa vector của {name}? Ảnh gốc vẫn giữ; train lại sẽ đăng ký lại người này."):
            self.stop_camera();self.database.remove(name);self.refresh_people()
            self.status.set(f"Đã xóa vector của {name}. Mở lại camera để cập nhật.")

    def start_camera(self):
        if self.busy():return
        from tkinter import messagebox
        if self.camera_thread and self.camera_thread.is_alive():
            self.status.set("Camera đang chạy hoặc đang dừng. Vui lòng đợi.");return
        try:
            index=int(self.camera_index.get());threshold=float(self.threshold.get());margin=float(self.margin.get())
            if not 0<=index<=20 or not .2<=threshold<=.99 or not 0<=margin<=.5:raise ValueError()
        except (ValueError,self.tk.TclError):
            messagebox.showerror("Thông số sai","Camera từ 0 đến 20; cosine 0.20–0.99; chênh lệch 0–0.50.");return
        try:
            (self.data/"settings.json").write_text(json.dumps({"threshold":threshold,"margin":margin}),encoding="utf-8")
        except OSError as exc:messagebox.showerror("Không lưu được cấu hình",str(exc));return
        self.stop_event=threading.Event()
        self.status.set("Đang chuẩn bị mô hình và mở camera...")
        self.camera_thread=threading.Thread(target=self.camera_worker,args=(index,threshold,margin,self.stop_event),daemon=True,name="camera")
        self.camera_thread.start()

    def camera_worker(self,index,threshold,margin,stop):
        capture=None
        try:
            ensure_models(self.models,self.log)
            if stop.is_set():return
            engine=FaceEngine(self.models);gallery=self.database.gallery();tracker=StableTracker()
            if not gallery:self.log("Database chua co vector: moi khuon mat se la unknown. Co the chup anh dang ky.")
            backends=[cv2.CAP_DSHOW,cv2.CAP_MSMF,cv2.CAP_ANY] if sys.platform=="win32" else [cv2.CAP_ANY]
            frame=None
            for backend in backends:
                if stop.is_set():return
                capture=cv2.VideoCapture(index,backend)
                capture.set(cv2.CAP_PROP_FRAME_WIDTH,960);capture.set(cv2.CAP_PROP_FRAME_HEIGHT,540)
                if capture.isOpened():
                    for _ in range(5):
                        ok,frame=capture.read()
                        if ok and frame is not None:break
                    if ok and frame is not None:break
                capture.release();capture=None
            if capture is None:raise RuntimeError("Không mở được camera. Đóng ứng dụng đang dùng camera, kiểm tra quyền camera hoặc thử chỉ số 1.")
            self.events.put(("status",f"Camera {index} đang bật • {len(gallery)} người • chỉ lưu ảnh khi bấm Chụp."))
            failures=0
            while not stop.is_set():
                started=time.monotonic()
                ok,frame=capture.read()
                if not ok or frame is None:
                    failures+=1
                    if failures>=8:raise RuntimeError("Mất kết nối camera. Tắt và mở lại camera.")
                    stop.wait(.05);continue
                failures=0
                frame=cv2.flip(frame,1)
                faces=engine.detect(frame);detections=[]
                valid_capture=False
                for face in faces:
                    valid,_=engine.quality(frame,face)
                    valid_capture=valid if len(faces)==1 else False
                    name,score="unknown",0.0
                    if valid:
                        name,score=match_identity(engine.feature(frame,face),gallery,threshold,margin)
                    detections.append((face[:4],name,score,valid))
                if stop.is_set():break
                with self.raw_lock:
                    self.raw_frame=frame.copy();self.capture_valid=valid_capture;self.raw_time=time.monotonic()
                results=tracker.update(detections,time.monotonic())
                display=draw_recognition(frame,results)
                try:self.frames.put_nowait(display)
                except queue.Full:
                    try:self.frames.get_nowait()
                    except queue.Empty:pass
                    try:self.frames.put_nowait(display)
                    except queue.Full:pass
                stop.wait(max(0,.065-(time.monotonic()-started)))
        except Exception as exc:
            self.events.put(("camera_error",str(exc)))
        finally:
            if capture is not None:capture.release()
            with self.raw_lock:self.raw_frame=None;self.capture_valid=False
            self.events.put(("camera_stopped",None))

    def stop_camera(self):
        self.stop_event.set()
        # UI never blocks on a slow camera driver. The worker releases it itself.
        self.status.set("Đang tắt camera...")
        with self.raw_lock:self.raw_frame=None;self.capture_valid=False

    def poll(self):
        from tkinter import messagebox
        if self.closed:return
        for _ in range(100):
            try:kind,value=self.events.get_nowait()
            except queue.Empty:break
            if kind=="log":
                self.log_view.configure(state="normal");self.log_view.insert("end",str(value)+"\n")
                if int(self.log_view.index("end-1c").split(".")[0])>600:self.log_view.delete("1.0","101.0")
                self.log_view.see("end");self.log_view.configure(state="disabled")
            elif kind in {"trained","train_error"}:
                self.training=False;self.progress.stop();self.train_button.state(["!disabled"]);self.start_button.state(["!disabled"])
                self.refresh_people()
                if kind=="trained":
                    self.status.set(f"Hoàn tất: {len(value['trained'])} người được cập nhật; {len(value['failed'])} người chưa đủ ảnh. Xem nhật ký.")
                    if not value["trained"]:messagebox.showwarning("Chưa có người được train","Không có người đủ ít nhất 3 ảnh hợp lệ. Xem lý do trong nhật ký.")
                else:self.status.set("Tạo vector chưa thành công.");messagebox.showerror("Lỗi tạo vector",value)
            elif kind=="status":
                if not self.training and not self.stop_event.is_set():self.status.set(value)
            elif kind=="camera_error":
                self.status.set(value);self.log(value);messagebox.showerror("Camera",value)
            elif kind=="camera_stopped":
                while True:
                    try:self.frames.get_nowait()
                    except queue.Empty:break
                self.preview.configure(image="",text="CAMERA ĐANG TẮT");self.preview.image=None
                if not self.training and self.status.get()=="Đang tắt camera...":self.status.set("Camera đã tắt.")
        try:
            image=self.frames.get_nowait()
            if not self.stop_event.is_set():
                image.thumbnail((max(320,self.preview.winfo_width()),max(240,self.preview.winfo_height())))
                photo=ImageTk.PhotoImage(image);self.preview.configure(image=photo,text="");self.preview.image=photo
        except queue.Empty:pass
        self.root.after(65,self.poll)

    def close(self):
        from tkinter import messagebox
        if self.training and not messagebox.askyesno("Đang train","Thoát khi đang tạo vector? Những người đã lưu thành công vẫn được giữ."):
            return
        self.closed=True;self.stop_event.set();self.root.destroy()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,default=BASE/"data")
    parser.add_argument("--train-dir",type=Path,default=BASE/"train")
    parser.add_argument("--camera",type=int,default=0)
    action=parser.add_mutually_exclusive_group()
    action.add_argument("--download-models",action="store_true")
    action.add_argument("--check",action="store_true")
    action.add_argument("--train",action="store_true")
    args=parser.parse_args()
    if args.check:
        print(f"Python {sys.version.split()[0]} | OpenCV {cv2.__version__} | NumPy {np.__version__}")
        print("YuNet API:",hasattr(cv2,"FaceDetectorYN"),"| SFace API:",hasattr(cv2,"FaceRecognizerSF"))
        for filename,_,checksum,size in MODELS.values():
            path=args.data_dir/"models"/filename
            ok=path.is_file() and path.stat().st_size==size and sha256(path)==checksum
            print(filename,": OK" if ok else ": chua co hoac khong hop le")
        return 0
    if args.download_models:
        ensure_models(args.data_dir/"models");return 0
    if args.train:
        ensure_models(args.data_dir/"models")
        report=enroll_directory(FaceEngine(args.data_dir/"models"),VectorDatabase(args.data_dir/"face_vectors.sqlite3"),args.train_dir)
        print(json.dumps(report,ensure_ascii=False,indent=2))
        return 0 if report["trained"] and not report["failed"] else 1
    import tkinter as tk
    root=tk.Tk()
    FaceApp(root,args.data_dir,args.train_dir,args.camera)
    root.mainloop()
    return 0


if __name__=="__main__":
    try:
        if hasattr(sys.stdout,"reconfigure"):sys.stdout.reconfigure(encoding="utf-8")
        sys.exit(main())
    except Exception as exc:
        print(f"LOI: {exc}",file=sys.stderr)
        sys.exit(1)
