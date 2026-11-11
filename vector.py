
from pathlib import Path
import os
import pandas as pd
import fitz

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


# =========================
# 1. CẤU HÌNH
# =========================

BASE_DIR = Path(__file__).resolve().parent

PDF_PATH = BASE_DIR / "GiaosTrinh_LTW_V_i_PHP1.pdf"
CSV_PATH = BASE_DIR / "reviews.csv"  # Đổi thành tên CSV thực tế

DB_LOCATION = BASE_DIR / "chroma_langchain_db"
COLLECTION_NAME = "qnu_ai_knowledge"

EMBEDDING_MODEL = "nomic-embed-text:latest"


# =========================
# 2. KHỞI TẠO EMBEDDING
# =========================

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)


# =========================
# 3. ĐỌC FILE PDF
# =========================

def load_pdf(file_path: Path) -> list[Document]:
    documents = []

    if not file_path.exists():
        print(f"Không tìm thấy PDF: {file_path}")
        return documents

    with fitz.open(file_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            content = page.get_text("text").strip()

            if content:
                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": file_path.name,
                            "type": "pdf",
                            "page": page_number,
                        },
                    )
                )

    print(f"PDF: đọc được {len(documents)} trang có văn bản.")
    return documents


# =========================
# 4. ĐỌC FILE CSV
# =========================

def load_csv(file_path: Path) -> list[Document]:
    documents = []

    if not file_path.exists():
        print(f"Không tìm thấy CSV: {file_path}")
        return documents

    # Chỉ thử các bảng mã, không giả định CSV nào cũng là UTF-8.
    df = None

    for encoding in ["utf-8-sig", "utf-8", "cp1258", "cp1252"]:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"CSV: đọc thành công bằng encoding={encoding}")
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        raise ValueError(
            f"Không xác định được bảng mã CSV: {file_path.name}"
        )

    # Chuyển từng hàng CSV thành một Document.
    for index, row in df.iterrows():
        fields = []

        for column, value in row.items():
            if pd.notna(value):
                fields.append(f"{column}: {value}")

        content = "\n".join(fields).strip()

        if content:
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": file_path.name,
                        "type": "csv",
                        "row": int(index),
                    },
                )
            )

    print(f"CSV: đọc được {len(documents)} hàng có dữ liệu.")
    return documents


# =========================
# 5. NẠP DỮ LIỆU VÀO CHROMA
# =========================

all_documents = []

all_documents.extend(load_pdf(PDF_PATH))
all_documents.extend(load_csv(CSV_PATH))

if not all_documents:
    raise ValueError(
        "Không có dữ liệu. Hãy kiểm tra đường dẫn và nội dung các file."
    )

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=str(DB_LOCATION),
    embedding_function=embeddings,
)


# =========================
# 6. THÊM DỮ LIỆU
# =========================

# Dùng ID xác định để tránh thêm trùng khi chạy lại.
ids = []

for index, document in enumerate(all_documents):
    source = document.metadata["source"]
    location = document.metadata.get(
        "page",
        document.metadata.get("row", index),
    )

    ids.append(f"{source}:{location}")

# Chỉ thêm tài liệu chưa tồn tại.
existing_ids = set()

for start in range(0, len(ids), 500):
    batch_ids = ids[start:start + 500]
    result = vector_store.get(ids=batch_ids)
    existing_ids.update(result["ids"])

new_documents = []
new_ids = []

for document, doc_id in zip(all_documents, ids):
    if doc_id not in existing_ids:
        new_documents.append(document)
        new_ids.append(doc_id)

if new_documents:
    vector_store.add_documents(
        documents=new_documents,
        ids=new_ids,
    )
    print(f"Đã thêm {len(new_documents)} tài liệu mới vào Chroma.")
else:
    print("Không có tài liệu mới cần thêm.")


# =========================
# 7. TẠO RETRIEVER
# =========================

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)

print("Retriever QNU-AI đã sẵn sàng.")
