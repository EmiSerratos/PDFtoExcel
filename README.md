# 📑 PDF → Excel (Consolidado)

Convierte múltiples **PDFs de boletines** en **un solo Excel** con estas columnas:  

- 🔢 **Numero Parte**  
- 🏷️ **EAN**  
- 🚗 **Aplicación Vehicular**  
- 💲 **Precio de Lista**  

---

## ⚙️ ¿Cómo funciona?

1. 📂 Subes uno o varios PDFs (boletines).  
2. 🤖 El sistema extrae la información de tablas y texto.  
3. 🧩 Consolida todo en **1 solo archivo Excel**.  
4. 📊 Obtienes una hoja **DATA** con tus columnas listas para usar.  

---

## 🚀 Despliegue en Hugging Face Spaces

1. Crea una cuenta en [Hugging Face](https://huggingface.co/).  
2. Ve a **New Space → Gradio**.  
3. Sube estos archivos a tu repo:  
   - `app.py`  
   - `requirements.txt`  
   - `apt.txt` *(solo si usarás OCR)*  
4. Espera a que se construya automáticamente.  
5. 🎉 ¡Listo! Ya tienes tu app web con un link para compartir con tu equipo.  

---

## 📄 Archivos clave

### requirements.txt
```txt
gradio
pdfplumber
pandas
openpyxl
pillow
pdf2image   # solo si usarás OCR
pytesseract # solo si usarás OCR
```

### apt.txt  *(solo si usarás OCR)*
```txt
tesseract-ocr
tesseract-ocr-spa
poppler-utils
```

---

## 🖥️ Uso

1. Abre la app en tu navegador (URL del Space).  
2. Arrastra tus **PDFs** al recuadro.  
3. (Opcional) Marca **Activar OCR** si tus PDFs son escaneados como imágenes.  
4. Clic en **Procesar y generar Excel**.  
5. Descarga tu archivo Excel consolidado ✅  

---

## 📝 Notas

- ⚡ Si los PDFs tienen texto seleccionable → no necesitas OCR.  
- 🐢 Si son escaneados, OCR funciona pero es más lento.  
- 🔍 Si cambian los encabezados de las tablas en el futuro, solo hay que ajustar el mapeo en `app.py`.  
- 🛡️ Los archivos no se guardan en el servidor, se procesan en memoria temporal.  
