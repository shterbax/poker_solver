"""
Poker Table ROI Calibrator Tool
===============================
Interactive desktop application to create and calibrate normalized ROI profiles
for poker solvers (supporting 6-max cash, 7-max MTT, or custom table layouts).
"""

import json
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk


class ROICalibratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Poker Table ROI Calibrator — Senior Solver Engine")
        self.root.geometry("1280x800")
        self.root.minsize(1024, 700)

        # State Variables
        self.image_path = None
        self.original_image = None
        self.display_image = None
        self.tk_image = None
        self.scale_factor = 1.0
        self.img_w = 0
        self.img_h = 0

        # Drawing state
        self.start_x = None
        self.start_y = None
        self.current_rect_id = None

        # Data structure
        self.table_type_var = tk.StringVar(value="6max_cash")
        self.profile_name_var = tk.StringVar(value="coinpoker_6max_cash")
        self.window_regex_var = tk.StringVar(value="^NLH.*")
        self.rois = {}  # key -> {"x": float, "y": float, "w": float, "h": float}

        self._build_ui()
        self._load_preset_targets()

    def _build_ui(self):
        # Top Toolbar
        toolbar = ttk.Frame(self.root, padding=5)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        btn_load_img = ttk.Button(toolbar, text="📷 Загрузить скриншот", command=self.load_image)
        btn_load_img.pack(side=tk.LEFT, padx=5)

        btn_load_json = ttk.Button(toolbar, text="📂 Открыть JSON профиль", command=self.load_json_profile)
        btn_load_json.pack(side=tk.LEFT, padx=5)

        btn_save_json = ttk.Button(toolbar, text="💾 Сохранить JSON профиль", command=self.save_json_profile)
        btn_save_json.pack(side=tk.LEFT, padx=5)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        ttk.Label(toolbar, text="Тип стола:").pack(side=tk.LEFT, padx=2)
        combo_type = ttk.Combobox(
            toolbar,
            textvariable=self.table_type_var,
            values=["6max_cash", "7max_mtt", "custom"],
            width=12,
            state="readonly"
        )
        combo_type.pack(side=tk.LEFT, padx=5)
        combo_type.bind("<<ComboboxSelected>>", self.on_table_type_change)

        ttk.Label(toolbar, text="Имя профиля:").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Entry(toolbar, textvariable=self.profile_name_var, width=22).pack(side=tk.LEFT, padx=5)

        # Main Layout
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left Frame: Control Panel
        left_frame = ttk.Frame(paned, width=320, padding=5)
        paned.add(left_frame, weight=0)

        ttk.Label(left_frame, text="Элементы разметки", font=("Helvetica", 11, "bold")).pack(anchor=tk.W, pady=(0, 5))

        self.target_listbox = tk.Listbox(left_frame, exportselection=False, font=("Consolas", 10))
        self.target_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        self.target_listbox.bind("<<ListboxSelect>>", self.on_target_select)

        self.info_frame = ttk.LabelFrame(left_frame, text="Координаты текущей зоны", padding=8)
        self.info_frame.pack(fill=tk.X, pady=5)

        self.lbl_norm_coords = ttk.Label(self.info_frame, text="X: -  Y: -\nW: -  H: -", font=("Consolas", 9))
        self.lbl_norm_coords.pack(anchor=tk.W)

        btn_clear_target = ttk.Button(self.info_frame, text="Очистить зону", command=self.clear_selected_roi)
        btn_clear_target.pack(anchor=tk.E, pady=(5, 0))

        # Right Frame: Canvas
        right_frame = ttk.Frame(paned, padding=5)
        paned.add(right_frame, weight=1)

        self.canvas = tk.Canvas(right_frame, bg="#1e1e1e", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.root.bind("<Configure>", self.on_window_resize)

    def _load_preset_targets(self):
        self.target_listbox.delete(0, tk.END)
        table_type = self.table_type_var.get()

        # Глобальные зоны
        targets = [
            "G: pot_amount",
            "G: board_cards",
            "G: hero_left",
            "G: hero_right"
        ]

        if table_type == "7max_mtt":
            targets.append("G: tournament_info")

        # Зоны игроков (включая локальный баттон и полосу таймера)
        seats_count = 6 if table_type == "6max_cash" else (7 if table_type == "7max_mtt" else 9)
        for i in range(seats_count):
            targets.extend([
                f"S{i}: stack",
                f"S{i}: bet",
                f"S{i}: timer_bar",   # <--- Цветовой индикатор текущего хода
                f"S{i}: status",      # Плашки "ЧЕК", "ФОЛД", "КОЛЛ"
                f"S{i}: dealer_btn"   # Локальный баттон D
            ])

        for t in targets:
            status_symbol = "✓ " if t in self.rois else "  "
            self.target_listbox.insert(tk.END, f"{status_symbol}{t}")

    def on_table_type_change(self, event=None):
        t_type = self.table_type_var.get()
        if t_type == "6max_cash":
            self.profile_name_var.set("coinpoker_6max_cash")
        elif t_type == "7max_mtt":
            self.profile_name_var.set("coinpoker_7max_mtt")
        self._load_preset_targets()
        self.redraw_all_rois()

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Выберите скриншот стола",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if not file_path:
            return

        self.image_path = file_path
        self.original_image = Image.open(file_path)
        self.img_w, self.img_h = self.original_image.size
        self.render_image()

    def render_image(self):
        if self.original_image is None:
            return

        canvas_w = max(self.canvas.winfo_width(), 100)
        canvas_h = max(self.canvas.winfo_height(), 100)

        scale_w = canvas_w / self.img_w
        scale_h = canvas_h / self.img_h
        self.scale_factor = min(scale_w, scale_h, 1.0)

        new_w = int(self.img_w * self.scale_factor)
        new_h = int(self.img_h * self.scale_factor)

        self.display_image = self.original_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(self.display_image)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.redraw_all_rois()

    def on_window_resize(self, event):
        if self.original_image:
            self.render_image()

    def get_selected_target_key(self):
        sel = self.target_listbox.curselection()
        if not sel:
            return None
        text = self.target_listbox.get(sel[0])
        return text.replace("✓ ", "").strip()

    def on_mouse_down(self, event):
        if not self.original_image:
            return
        self.start_x = event.x
        self.start_y = event.y
        if self.current_rect_id:
            self.canvas.delete(self.current_rect_id)
        self.current_rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline="#00FF00", width=2
        )

    def on_mouse_drag(self, event):
        if not self.start_x or not self.original_image:
            return
        self.canvas.coords(self.current_rect_id, self.start_x, self.start_y, event.x, event.y)

    def on_mouse_up(self, event):
        if not self.start_x or not self.original_image:
            return

        end_x, end_y = event.x, event.y
        x1, x2 = min(self.start_x, end_x), max(self.start_x, end_x)
        y1, y2 = min(self.start_y, end_y), max(self.start_y, end_y)

        if (x2 - x1) < 5 or (y2 - y1) < 5:
            self.canvas.delete(self.current_rect_id)
            self.current_rect_id = None
            return

        target_key = self.get_selected_target_key()
        if not target_key:
            messagebox.showwarning("Предупреждение", "Сначала выберите элемент из списка слева!")
            self.canvas.delete(self.current_rect_id)
            self.current_rect_id = None
            return

        img_px_x1 = x1 / self.scale_factor
        img_px_y1 = y1 / self.scale_factor
        img_px_w = (x2 - x1) / self.scale_factor
        img_px_h = (y2 - y1) / self.scale_factor

        norm_x = round(img_px_x1 / self.img_w, 4)
        norm_y = round(img_px_y1 / self.img_h, 4)
        norm_w = round(img_px_w / self.img_w, 4)
        norm_h = round(img_px_h / self.img_h, 4)

        self.rois[target_key] = {
            "x": norm_x,
            "y": norm_y,
            "w": norm_w,
            "h": norm_h
        }

        self.update_info_label(norm_x, norm_y, norm_w, norm_h)
        self.update_listbox_item_status(target_key, True)
        self.redraw_all_rois()

    def update_info_label(self, x, y, w, h):
        self.lbl_norm_coords.config(
            text=f"X: {x:.4f}  Y: {y:.4f}\nW: {w:.4f}  H: {h:.4f}"
        )

    def update_listbox_item_status(self, key, done: bool):
        items = self.target_listbox.get(0, tk.END)
        for idx, item in enumerate(items):
            clean = item.replace("✓ ", "").strip()
            if clean == key:
                prefix = "✓ " if done else "  "
                self.target_listbox.delete(idx)
                self.target_listbox.insert(idx, f"{prefix}{clean}")
                self.target_listbox.selection_set(idx)
                break

    def on_target_select(self, event):
        target_key = self.get_selected_target_key()
        if target_key and target_key in self.rois:
            roi = self.rois[target_key]
            self.update_info_label(roi["x"], roi["y"], roi["w"], roi["h"])
        else:
            self.lbl_norm_coords.config(text="X: -  Y: -\nW: -  H: -")
        self.redraw_all_rois()

    def clear_selected_roi(self):
        target_key = self.get_selected_target_key()
        if target_key and target_key in self.rois:
            del self.rois[target_key]
            self.update_listbox_item_status(target_key, False)
            self.redraw_all_rois()

    def redraw_all_rois(self):
        if not self.original_image:
            return

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        selected_key = self.get_selected_target_key()

        for key, roi in self.rois.items():
            px_x = int(roi["x"] * self.img_w * self.scale_factor)
            px_y = int(roi["y"] * self.img_h * self.scale_factor)
            px_w = int(roi["w"] * self.img_w * self.scale_factor)
            px_h = int(roi["h"] * self.img_h * self.scale_factor)

            is_selected = (key == selected_key)
            color = "#00FF00" if is_selected else "#00BFFF"
            width = 3 if is_selected else 1

            self.canvas.create_rectangle(
                px_x, px_y, px_x + px_w, px_y + px_h,
                outline=color, width=width
            )

            self.canvas.create_text(
                px_x + 2, px_y - 8, anchor=tk.NW,
                text=key, fill=color, font=("Consolas", 8, "bold")
            )

    def load_json_profile(self):
        file_path = filedialog.askopenfilename(
            title="Открыть JSON профиль",
            filetypes=[("JSON files", "*.json")]
        )
        if not file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.profile_name_var.set(data.get("profile_name", "custom_profile"))
            self.table_type_var.set(data.get("table_type", "6max_cash"))
            self.window_regex_var.set(data.get("window_title_regex", "^NLH.*"))

            # Парсим чистые данные из global_rois и seats во внутренний формат GUI
            loaded_rois = {}

            # 1. Читаем глобальные зоны
            global_rois = data.get("global_rois", {})
            for g_name, coords in global_rois.items():
                key = f"G: {g_name}" if not g_name.startswith("G: ") else g_name
                loaded_rois[key] = coords

            # 2. Читаем зоны игроков
            seats = data.get("seats", [])
            for seat in seats:
                seat_id = seat.get("seat_id")
                if seat_id is None:
                    continue
                for field_name, coords in seat.items():
                    if field_name == "seat_id":
                        continue
                    key = f"S{seat_id}: {field_name}"
                    loaded_rois[key] = coords

            # Фолбэк на случай старых/нестандартных файлов
            if not loaded_rois:
                loaded_rois = data.get("raw_flat_rois", data.get("rois", {}))

            self.rois = loaded_rois

            # Перестраиваем список с учетом загруженных галочек
            self._load_preset_targets()
            self.redraw_all_rois()

            messagebox.showinfo("Успех",
                                f"Профиль '{self.profile_name_var.get()}' успешно загружен! Загружено зон: {len(self.rois)}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить профиль:\n{e}")

    def save_json_profile(self):
        if not self.rois:
            messagebox.showwarning("Предупреждение", "Нет размеченных зон для сохранения!")
            return

        file_path = filedialog.asksaveasfilename(
            title="Сохранить JSON профиль",
            defaultextension=".json",
            initialfile=f"{self.profile_name_var.get()}.json",
            filetypes=[("JSON files", "*.json")]
        )
        if not file_path:
            return

        global_rois = {}
        seats_rois = []

        table_type = self.table_type_var.get()
        seats_count = 6 if table_type == "6max_cash" else (7 if table_type == "7max_mtt" else 9)

        for i in range(seats_count):
            seats_rois.append({"seat_id": i})

        for key, roi in self.rois.items():
            if key.startswith("G: "):
                clean_name = key.replace("G: ", "")
                global_rois[clean_name] = roi
            elif key.startswith("S"):
                parts = key.split(":")
                seat_idx = int(parts[0].replace("S", ""))
                field_name = parts[1].strip()
                if seat_idx < len(seats_rois):
                    seats_rois[seat_idx][field_name] = roi

        # Итоговая структура JSON без raw_flat_rois
        export_data = {
            "profile_name": self.profile_name_var.get(),
            "table_type": self.table_type_var.get(),
            "window_title_regex": self.window_regex_var.get(),
            "max_seats": seats_count,
            "global_rois": global_rois,
            "seats": seats_rois
        }

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            messagebox.showinfo("Успех", f"Профиль успешно сохранен в:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл:\n{e}")


def main():
    root = tk.Tk()
    app = ROICalibratorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()