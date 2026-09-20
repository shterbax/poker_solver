import cv2
import numpy as np
from pathlib import Path
from typing import Optional

try:
    import win32gui
    import ctypes
    from ctypes import wintypes
    import mss

    HAS_WIN32_MSS = True
except ImportError:
    HAS_WIN32_MSS = False


def capture_and_save_coinpoker_window(output_filename: str = "test_table.png") -> Optional[np.ndarray]:
    if not HAS_WIN32_MSS:
        print("❌ Установите зависимости: pip install pywin32 mss")
        return None

    target_hwnd = None

    def callback(handle, extra):
        nonlocal target_hwnd
        if win32gui.IsWindowVisible(handle) and not win32gui.IsIconic(handle):
            title = win32gui.GetWindowText(handle)
            if title and "coinpoker" in title.lower():
                # Проверяем размеры окна, чтобы исключить мини-окна и иконки
                rect = win32gui.GetWindowRect(handle)
                w = rect[2] - rect[0]
                h = rect[3] - rect[1]

                # Игровой стол CoinPoker имеет характерный крупный размер (больше 700x500)
                if w > 700 and h > 500:
                    target_hwnd = handle

    win32gui.EnumWindows(callback, None)

    if not target_hwnd:
        print("❌ Окно стола CoinPoker не найдено. Убедитесь, что стол открыт на экране.")
        return None

    # Получаем точные границы найденного стола через DWM
    rect = wintypes.RECT()
    try:
        DWMWA_EXTENDED_FRAME_BOUNDS = 9
        ctypes.windll.dwmapi.DwmGetWindowAttribute(
            target_hwnd,
            ctypes.wintypes.DWORD(DWMWA_EXTENDED_FRAME_BOUNDS),
            ctypes.byref(rect),
            ctypes.sizeof(rect)
        )
        left, top, right, bottom = rect.left, rect.top, rect.right, rect.bottom
    except Exception:
        left, top, right, bottom = win32gui.GetWindowRect(target_hwnd)

    width = right - left
    height = bottom - top

    # Захват через mss
    try:
        with mss.MSS() as sct:
            monitor = {"top": top, "left": left, "width": width, "height": height}
            sct_img = sct.grab(monitor)
            frame = np.array(sct_img)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            project_root = Path(__file__).resolve().parent
            if project_root.name == "tests":
                project_root = project_root.parent

            tests_dir = project_root / "tests"
            tests_dir.mkdir(parents=True, exist_ok=True)

            save_path = tests_dir / output_filename
            cv2.imwrite(str(save_path), frame_bgr)
            print(f"💾 Скриншот стола успешно сохранен: {save_path.resolve()}")

            return frame_bgr
    except Exception as e:
        print(f"❌ Ошибка захвата: {e}")
        return None


if __name__ == "__main__":
    print("=== ЗАПУСК ТЕСТА ЗАХВАТА ===")
    result = capture_and_save_coinpoker_window()
    if result is not None:
        print("=== ТЕСТ УСПЕШНО ЗАВЕРШЕН ===")
    else:
        print("=== ТЕСТ ЗАВЕРШЕН С ОШИБКОЙ ===")