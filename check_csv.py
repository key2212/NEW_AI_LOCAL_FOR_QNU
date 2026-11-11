
from pathlib import Path

file_path = Path("GiaosTrinh_LTW_V_i_PHP1.csv")
data = file_path.read_bytes()

print("Kích thước file:", len(data), "bytes")
print("BOM:", data[:4])

for encoding in [
    "utf-8",
    "utf-8-sig",
    "cp1258",
    "cp1252",
    "cp1250",
    "latin1"
]:
    try:
        text = data.decode(encoding)
        print(f"\n--- {encoding}: giải mã thành công ---")
        print(text[:500])
    except UnicodeDecodeError as e:
        print(f"{encoding}: thất bại tại byte {e.start}")
