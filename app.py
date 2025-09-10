# app.py
import io
import os
import re
import tempfile
from typing import List, Tuple, Dict, Any, Optional

import gradio as gr
import pandas as pd
import pdfplumber

# OCR opcional (requieres pdf2image y pytesseract + tesseract/poppler en el sistema)
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# =========================
# Configuración de destino
# =========================
TARGET_COLUMNS = ["Numero Parte", "EAN", "Aplicación Vehicular", "Precio de Lista"]

# Nombres de columnas esperadas (tablas) -> campos de destino
# (normalizamos encabezados a upper y sin acentos/espacios para comparar)
HEADERS_MAP = {
    "NUMEROPARTE": "Numero Parte",
    "NOPARTE": "Numero Parte",
    "NºPARTE": "Numero Parte",
    "N°PARTE": "Numero Parte",
    "PARTE": "Numero Parte",

    "EAN": "EAN",
    "CODIGOEAN": "EAN",

    "PRECIODELISTA": "Precio de Lista",
    "PRECIO": "Precio de Lista",
    "PRECIO$": "Precio de Lista",

    # columnas que forman la "Aplicación Vehicular" cuando existan
    "ARMADORA": "Aplicación Vehicular",
    "MODELO": "Aplicación Vehicular",
    "LITROS": "Aplicación Vehicular",
    "AÑOS": "Aplicación Vehicular",
    "ANOS": "Aplicación Vehicular",
    "TRANSMISIÓN": "Aplicación Vehicular",
    "TRANSMISION": "Aplicación Vehicular",
    "POSICIÓN": "Aplicación Vehicular",
    "POSICION": "Aplicación Vehicular",
}

APP_COMPONENT_KEYS = {"ARMADORA","MODELO","LITROS","AÑOS","ANOS","TRANSMISIÓN","TRANSMISION","POSICIÓN","POSICION"}


# =========================
# Utilidades
# =========================
def normalize_header(s: str) -> str:
    if s is None: return ""
    s = str(s)
    s = s.strip()
    # quitar acentos
    tr = str.maketrans("ÁÉÍÓÚÜÑáéíóúüñ", "AEIOUUNAEIOUUN")
    s = s.translate(tr)
    s = re.sub(r"\s+", "", s.upper())
    return s

def clean_text(s: Optional[str]) -> str:
    if s is None: return ""
    s = str(s).strip()
    s = s.replace("\n", " ")
    return re.sub(r"\s+", " ", s)

def find_price_in_text(s: str) -> Optional[str]:
    # Busca $ 1,234.56 o $1234,56 etc.
    m = re.search(r"\$\s*([\d\.,]+)", s)
    return f"${m.group(1)}" if m else None

def find_ean_in_text(s: str) -> Optional[str]:
    # 13 dígitos típicos de EAN
    m = re.search(r"\b(\d{13})\b", s)
    return m.group(1) if m else None

def looks_like_part(s: str) -> bool:
    # Heurística básica para detectar número de parte
    # ejemplos: "7082 STD", "RR-9607", "M2031", "T1331"
    return bool(re.match(r"^[A-Z0-9\-_/]+(?:\s+[A-Z0-9\-_/]+)?$", s.strip()))

def compose_app_from_row(row: Dict[str, Any]) -> str:
    parts = []
    for k in ["Armadora", "Modelo", "Litros", "Años", "Transmisión", "Posición"]:
        v = clean_text(row.get(k))
        if v:
            parts.append(v)
    return " ".join(parts).strip()


# =========================
# Extracción desde tablas
# =========================
def extract_rows_from_table(df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Intenta mapear una tabla a filas con los 4 campos objetivo.
    - Si existen columnas separadas para Armadora/Modelo/... compone Aplicación Vehicular.
    - Si hay una sola columna de 'Aplicación', úsala tal cual.
    """
    if df is None or df.empty:
        return []

    # normaliza encabezados
    df = df.copy()
    df.columns = [clean_text(c) for c in df.columns]
    header_norm = [normalize_header(c) for c in df.columns]

    # Mapeo de columnas -> destino
    col_to_dest = {}
    for i, hn in enumerate(header_norm):
        if hn in HEADERS_MAP:
            col_to_dest[i] = HEADERS_MAP[hn]

    # Detectar si podemos componer "Aplicación Vehicular" por columnas
    app_components_present = any(normalize_header(c) in APP_COMPONENT_KEYS for c in df.columns)

    results = []
    for _, row in df.iterrows():
        rec = {k: "" for k in TARGET_COLUMNS}
        # 1) recorre columnas mapeadas directas
        for i, dest in col_to_dest.items():
            val = clean_text(row.iloc[i]) if i < len(row) else ""
            # si es precio y no está formateado, intenta formatearlo
            if dest == "Precio de Lista" and val and not val.strip().startswith("$"):
                # busca $ en toda la fila como respaldo
                p2 = find_price_in_text(" ".join([clean_text(str(x)) for x in row.values]))
                val = p2 or val
            rec[dest] = val

        # 2) compone Aplicación si hay componentes distribuidos
        if app_components_present:
            # crea dict con nombres legibles
            pretty = {clean_text(df.columns[i]): clean_text(row.iloc[i]) for i in range(len(df.columns))}
            app = compose_app_from_row(pretty)
            if app:
                rec["Aplicación Vehicular"] = app

        # 3) si no hay Numero Parte, intenta inferir del primer campo
        if not rec["Numero Parte"]:
            first_cell = clean_text(row.iloc[0]) if len(row) else ""
            if looks_like_part(first_cell):
                rec["Numero Parte"] = first_cell

        # 4) si no hay EAN aún, búscalo en toda la fila
        if not rec["EAN"]:
            ean = find_ean_in_text(" ".join([clean_text(str(x)) for x in row.values]))
            if ean:
                rec["EAN"] = ean

        # 5) si no hay Precio aún, intenta buscar en toda la fila
        if not rec["Precio de Lista"]:
            pr = find_price_in_text(" ".join([clean_text(str(x)) for x in row.values]))
            if pr:
                rec["Precio de Lista"] = pr

        # descarta filas vacías
        if any(rec.values()):
            results.append(rec)

    return results


# =========================
# Extracción desde texto
# =========================
def extract_rows_from_text(lines: List[str]) -> List[Dict[str, str]]:
    """
    Heurística para boletines sin tablas:
    - Agrupa por "bloques" cuando detecta un posible número de parte.
    - Dentro del bloque busca EAN: XXXX y Precio de Lista ($ ...)
    - La aplicación vehicular se intenta armar con líneas vecinas si hay pistas.
    """
    rows = []
    current = {"Numero Parte": "", "EAN": "", "Aplicación Vehicular": "", "Precio de Lista": ""}

    def flush():
        if any(current.values()):
            rows.append(current.copy())
        # resetea
        current.update({k: "" for k in current.keys()})

    # patrones
    part_pat = re.compile(r"^(?:N[°º]?\s*PARTE[:#]?\s*)?([A-Z0-9][A-Z0-9\-_\/]*?(?:\s+[A-Z0-9\-_\/]+)?)\s*$")
    ean_pat = re.compile(r"\bEAN[: ]+(\d{13})\b", re.IGNORECASE)
    price_line_pat = re.compile(r"(?:PRECIO\s*DE\s*LISTA[: ]+|\$)\s*([\d\.,]+)", re.IGNORECASE)

    for ln in lines:
        line = clean_text(ln)
        if not line:
            continue

        # Detecta nueva parte (o encabezado tipo "N° Parte: XXX" o una línea que parece parte)
        m_part = ean_header = None
        # Evita que EAN o precio se confundan como parte
        if not ean_pat.search(line) and not price_line_pat.search(line):
            m_part = part_pat.match(line)
        if m_part and looks_like_part(m_part.group(1)):
            # si ya había datos en curso, flush
            if any(current.values()):
                flush()
            current["Numero Parte"] = m_part.group(1).strip()
            continue

        # EAN
        m_ean = ean_pat.search(line)
        if m_ean:
            current["EAN"] = m_ean.group(1).strip()

        # Precio
        m_pr = price_line_pat.search(line)
        if m_pr:
            current["Precio de Lista"] = f"${m_pr.group(1)}"

        # Intento simple para aplicación (si hay pistas como Armadora/Modelo/Litros/Años/Transmisión/Posición)
        if any(key in line.upper() for key in ["ARMADORA", "MODELO", "LITROS", "AÑOS", "ANOS", "TRANSMISION", "TRANSMISIÓN", "POSICION", "POSICIÓN"]):
            # guarda la línea completa como parte de la aplicación
            app_prev = current["Aplicación Vehicular"]
            sep = " | " if app_prev else ""
            current["Aplicación Vehicular"] = app_prev + sep + line

    # último bloque
    if any(current.values()):
        flush()

    # Filtra filas sin Numero Parte y sin EAN ni Precio
    rows = [r for r in rows if r["Numero Parte"] or r["EAN"] or r["Precio de Lista"]]
    return rows


# =========================
# Pipeline por PDF
# =========================
def process_pdf_to_records(file_bytes: bytes, use_ocr: bool) -> List[Dict[str, str]]:
    """
    Devuelve lista de registros con TARGET_COLUMNS para un PDF.
    1) Intenta tablas. 2) Si no hay tablas válidas, intenta texto (nativo u OCR).
    """
    records: List[Dict[str, str]] = []

    # 1) Tablas
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            # Intento 1 (lines)
            tables = page.extract_tables(table_settings={
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "intersection_x_tolerance": 5,
                "intersection_y_tolerance": 5,
            }) or []
            # Intento 2 (stream)
            if not tables:
                tables = page.extract_tables(table_settings={
                    "vertical_strategy": "text",
                    "horizontal_strategy": "text",
                    "snap_tolerance": 3,
                    "join_tolerance": 3,
                    "edge_min_length": 3,
                }) or []
            for t in tables:
                df = pd.DataFrame(t)
                df.replace("\n", " ", regex=False, inplace=True)
                df = df.dropna(how="all").dropna(axis=1, how="all")
                if df.shape[0] == 0:
                    continue
                # usa primera fila como encabezado si parece cabecera
                if df.shape[0] > 1 and df.iloc[0].isnull().sum() < (df.shape[1] // 2):
                    df.columns = [str(c).strip() if pd.notna(c) else f"col_{i+1}" for i, c in enumerate(df.iloc[0])]
                    df = df.iloc[1:].reset_index(drop=True)
                recs = extract_rows_from_table(df)
                records.extend(recs)

    # 2) Texto si no hubo tablas o no se obtuvieron registros suficientes
    if not records:
        # Texto nativo
        text_lines = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                tx = page.extract_text() or ""
                for ln in tx.splitlines():
                    if ln.strip():
                        text_lines.append(ln)

        # ¿OCR?
        if use_ocr and not text_lines:
            if OCR_AVAILABLE:
                images = convert_from_bytes(file_bytes, fmt="png")
                for img in images:
                    tx = pytesseract.image_to_string(img, lang="spa+eng")
                    for ln in tx.splitlines():
                        if ln.strip():
                            text_lines.append(ln)

        if text_lines:
            recs = extract_rows_from_text(text_lines)
            records.extend(recs)

    # Normaliza a columnas objetivo, formatea valores
    out = []
    for r in records:
        row = {k: clean_text(r.get(k, "")) for k in TARGET_COLUMNS}
        out.append(row)

    # Dedup básico
    if out:
        df = pd.DataFrame(out)
        df = df.drop_duplicates(subset=["Numero Parte", "EAN", "Aplicación Vehicular", "Precio de Lista"], keep="first")
        out = df.to_dict(orient="records")

    return out


# =========================
# Excel consolidado
# =========================
def build_single_excel(files: List[Tuple[str, bytes]], use_ocr: bool) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        xlsx_path = os.path.join(tmpdir, "consolidado_pdf_a_excel.xlsx")

        all_rows = []
        for fname, fbytes in files:
            recs = process_pdf_to_records(fbytes, use_ocr=use_ocr)
            for r in recs:
                r["_archivo_origen"] = os.path.basename(fname)
            all_rows.extend(recs)

        df_out = pd.DataFrame(all_rows, columns=TARGET_COLUMNS + ["_archivo_origen"])

        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            # DATA (lo que te interesa)
            df_out[TARGET_COLUMNS].to_excel(writer, index=False, sheet_name="DATA")
            # OPCIONAL: hoja de trazabilidad
            df_out.to_excel(writer, index=False, sheet_name="DATA_con_origen")

        # Persistir a /tmp para descarga en Gradio
        dst = os.path.join("/tmp", os.path.basename(xlsx_path))
        with open(xlsx_path, "rb") as src, open(dst, "wb") as out:
            out.write(src.read())
        return dst


# =========================
# Interfaz Gradio
# =========================
TITLE = "PDF → Excel (Numero Parte, EAN, Aplicación, Precio)"
DESC = (
    "Sube múltiples PDFs y obtén un Excel con columnas: "
    "'Numero Parte', 'EAN', 'Aplicación Vehicular', 'Precio de Lista'. "
    "Activa OCR si tus PDFs son escaneados (imágenes)."
)

with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(DESC)

    with gr.Row():
        files_in = gr.File(label="Arrastra tus PDFs (múltiples)", file_count="multiple", file_types=[".pdf"])
    with gr.Row():
        use_ocr = gr.Checkbox(label="Activar OCR (para PDFs escaneados)", value=False)
    with gr.Row():
        run_btn = gr.Button("Procesar y generar Excel")
    with gr.Row():
        out_excel = gr.File(label="Descargar Excel")

    def _run(files, use_ocr_flag):
        if not files:
            return None
        file_tuples = []
        for f in files:
            with open(f.name, "rb") as fh:
                file_tuples.append((os.path.basename(f.name), fh.read()))
        excel_path = build_single_excel(file_tuples, bool(use_ocr_flag))
        return excel_path

    run_btn.click(_run, inputs=[files_in, use_ocr], outputs=[out_excel])

if __name__ == "__main__":
    demo.launch()
