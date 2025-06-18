import streamlit as st
import numpy as np
import pandas as pd
from paddleocr import PaddleOCR
import cv2
from pdf2image import convert_from_bytes
from PIL import Image
import subprocess
import json
import re
import time
from datetime import datetime
import requests

ocr = PaddleOCR(use_angle_cls=True, lang='en')

def preprocess_image(pil_img):
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(thresh)

def extract_text_paddleocr(_images):
    all_text = []
    for img in _images:
        preprocessed = preprocess_image(img)
        result = ocr.ocr(np.array(preprocessed), cls=True)
        page_text = "".join([line[1][0] for line in result[0]])
        all_text.append(page_text)
    return "".join(all_text).strip()

def run_llama_cached(prompt: str, model_name="qwen2.5:7b"):
    try:
        result = subprocess.run(
            ['ollama', 'run', model_name],
            input=prompt,
            capture_output=True,
            text=True,
            encoding='utf-8',
        )
        return result.stdout.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"

def extract_json_from_text(text):
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        try:
            obj, end = decoder.raw_decode(text[idx:].lstrip())
            return obj
        except json.JSONDecodeError:
            idx += 1
    return None

def create_kpi_prompt(invoice_text):
    return f"""
You are an invoice parser AI.

Your task is to extract structured data from the provided invoice text and return ONLY a JSON object with the following exact keys:

- "Company Name"
- "Purchase Order Number" (usually starts with 'PO'; do NOT include terms like "dtd.")
- "Product" (list of pharmaceutical ingredient names only; exclude and ingnore anything with non-medical descriptors like "fiber", "container", "drum", "bottle", etc.; keep only pure medicine names such as "Nimesulide", "Paracetamol", etc.; do not include packaging/counts like drums, fiber drums, bottles, etc.)
- "Quantity" (list; consider lines with phrasing "Kg")
- "Price per Kg" (list; consider lines with phrasing "/Kg")
- "Dispatch date" (format: DD/MM/YYYY; extracted from buyer's order number and date; "")
- "Conditions" (FOB or CIF only and exclude any extra text; from terms of delivery and payment)
- "Status"
- "Party Invoice Number" (from Invoice No. & Date; Exclude the date)
- "Party Invoice Date" (from Invoice No. & Date; Exclude the Invoice Number; Usually present with Party Invoice Number)
- "Docuemnt Invoice Date" (usually in the line with 'PO'; after the term like "dtd." or "Buyer's Order Number & Date"; Usually with Purchase Order Number)

Requirements:
- Return ONLY a valid JSON object (no extra text, explanation, or comments).
- JSON must be properly formatted.
- Lists for Product, Quantity, and Price per Kg MUST be of the same length.

Use the invoice text below as input:
---
{invoice_text}
---
Output ONLY the JSON object, with no introductory or trailing text.
"""

def clean_numeric(text):
    # Remove any non-numeric characters except decimal points
    return re.sub(r"[^\d.]", "", str(text))

# ----------- Streamlit UI -----------

st.set_page_config(page_title="AI Invoice Analyzer", layout="wide")
st.title("📄 AI Invoice Analyzer")

with st.sidebar:
    st.markdown("## 📎 Upload PDF")
    uploaded_pdf = st.file_uploader("Invoice PDF", type=["pdf"])
    if "run_pipeline" not in st.session_state:
        st.session_state.run_pipeline = False
    if "ocr_text" not in st.session_state:
        st.session_state.ocr_text = ""
    if "parsed_json" not in st.session_state:
        st.session_state.parsed_json = {}
    run_button = st.button("Run OCR & Extract Details")
    if run_button:
        if uploaded_pdf:
            st.session_state.run_pipeline = True
        else:
            st.warning("Please upload a PDF before running OCR.")

processComplete = False

if uploaded_pdf:
    file_bytes = uploaded_pdf.read()
    images = convert_from_bytes(file_bytes)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🖼️ PDF Preview")
        for i, img in enumerate(images):
            st.image(img, caption=f"Page {i+1}", use_container_width=True)

    with col2:
        if st.session_state.run_pipeline:
            with st.spinner("🔍 Running OCR..."):
                ocr_start = time.time()
                st.session_state.ocr_text = extract_text_paddleocr(images)
                ocr_time = time.time() - ocr_start

            with st.expander("📄 Show OCR Text"):
                st.text_area("OCR Output", value=st.session_state.ocr_text, height=150, disabled=True)

            st.subheader("🧠 Extracted KPI JSON")
            prompt = create_kpi_prompt(st.session_state.ocr_text)
            with st.spinner("💬 Running LLaMA..."):
                llama_start = time.time()
                llama_output = run_llama_cached(prompt)
                llama_time = time.time() - llama_start

            st.session_state.parsed_json = extract_json_from_text(llama_output)
            if st.session_state.parsed_json:
                st.success("✅ JSON extracted successfully")
            else:
                st.error("❌ Failed to extract JSON")
                st.code(llama_output)
                st.stop()
            st.info(f"🕒 OCR Time: {ocr_time:.2f}s | LLaMA Time: {llama_time:.2f}s")
            st.session_state.run_pipeline = False
        else:
            if st.session_state.ocr_text:
                with st.expander("📄 Show OCR Text"):
                    st.text_area("OCR Output", value=st.session_state.ocr_text, height=150, disabled=True)
            if st.session_state.parsed_json:
                st.subheader("🧠 Extracted KPI JSON")
                st.success("✅ Using cached extracted JSON")

    parsed_json = st.session_state.parsed_json
    st.markdown("### ✏️ Editable Extracted Data")
    if isinstance(parsed_json, str):
        try:
            parsed_json = json.loads(parsed_json)
        except json.JSONDecodeError:
            st.error("❌ Parsed JSON is not valid. Please re-run extraction.")
            parsed_json = {}

    def listify(x): 
        return x if isinstance(x, list) else [x]

    parsed_json["Company Name"] = st.text_input("Company Name", parsed_json.get("Company Name", ""))
    parsed_json["Purchase Order Number"] = st.text_input("Purchase Order Number", parsed_json.get("Purchase Order Number", ""))

    # Product
    products = listify(parsed_json.get("Product", []))
    quantities = listify(parsed_json.get("Quantity", []))
    prices = listify(parsed_json.get("Price per Kg", []))

    max_len = max(len(products), len(quantities), len(prices))
    products += [""] * (max_len - len(products))
    quantities += [""] * (max_len - len(quantities))
    prices += [""] * (max_len - len(prices))

    quantities = [clean_numeric(q) for q in quantities]
    prices = [clean_numeric(p) for p in prices]

    st.markdown("### 📃 Product List")
    df = pd.DataFrame({
        "Product": products,
        "Quantity": quantities,
        "Price per Kg": prices
    })
    edited_df = st.data_editor(df, num_rows="dynamic")

    parsed_json["Product"] = edited_df["Product"].tolist()
    parsed_json["Quantity"] = edited_df["Quantity"].tolist()
    parsed_json["Price per Kg"] = edited_df["Price per Kg"].tolist()

    # Dispatch Date
    try:
        dispatch_date = datetime.strptime(parsed_json.get("Dispatch date", ""), "%d/%m/%Y").date()
    except:
        dispatch_date = datetime.today().date()
    dispatch_date = st.date_input("Dispatch Date (YYYY/MM/DD)", dispatch_date)
    parsed_json["Dispatch date"] = dispatch_date.strftime("%d/%m/%Y")

    # Condition
    raw_conditions = parsed_json.get("Conditions", "").lower()
    if "fob" in raw_conditions:
        clean_condition = "FOB"
    elif "cif" in raw_conditions:
        clean_condition = "CIF"
    else:
        clean_condition = parsed_json.get("Conditions", "")
    parsed_json["Conditions"] = st.text_input("Conditions", clean_condition)

    dispatch_status_options = ["Dispatched", "Airway Pending", "BL Pending", "Under Dispatch"]
    parsed_json["Dispatch Status"] = st.selectbox("Dispatch Status", dispatch_status_options, index=dispatch_status_options.index(parsed_json.get("Dispatch Status", dispatch_status_options[0])))

    # Airway Bill Date
    try:
        airwaybill_date = datetime.strptime(parsed_json.get("Airway Bill Date", ""), "%d/%m/%Y").date()
    except:
        airwaybill_date = dispatch_date
    airwaybill_date = st.date_input("Airway Bill Date (YYYY/MM/DD)", value=airwaybill_date)
    parsed_json["Airway Bill Date"] = airwaybill_date.strftime("%d/%m/%Y")

    # Invoice Due Date
    add_days = st.number_input("📅 'X' days from Airway Bill Date (DAYS)", min_value=0, value=0, step=1)
    invoice_due_date = airwaybill_date + pd.Timedelta(days=add_days)
    invoice_due_date = st.date_input("Invoice Due Date (YYYY/MM/DD)", invoice_due_date)
    parsed_json["Invoice Due Date"] = invoice_due_date.strftime("%d/%m/%Y")

    # Party Invoice Number
    parsed_json["Party Invoice Number"] = st.text_input("Party Invoice Number", parsed_json.get("Party Invoice Number", ""))

    # Invoice Date
    try:
        invoice_date = datetime.strptime(parsed_json.get("Party Invoice Date", ""), "%d/%m/%Y").date()
    except:
        invoice_date = dispatch_date
    invoice_date = st.date_input("Party Invoice Date (YYYY/MM/DD)", value=invoice_date)
    parsed_json["Party Invoice Date"] = invoice_date.strftime("%d/%m/%Y")

    # Purchase Order Date
    try:
        po_date = datetime.strptime(parsed_json.get("Docuemnt Invoice Date", ""), "%d/%m/%Y").date()
    except:
        po_date = dispatch_date
    po_date = st.date_input("Purchase Order Date (YYYY/MM/DD)", value=po_date)
    parsed_json["Docuemnt Invoice Date"] = po_date.strftime("%d/%m/%Y")

    processComplete = True

else:
    parsed_json = {}

if processComplete:
    # 🧾 Create row-wise table from `final_op`
    flattened_rows = []
    final_op = parsed_json

    products = final_op.get("Product", [])
    quantities = final_op.get("Quantity", [])
    prices = final_op.get("Price per Kg", [])

    max_len = max(len(products), len(quantities), len(prices))

    # Fill missing with blanks to align lengths
    products += [""] * (max_len - len(products))
    quantities += [""] * (max_len - len(quantities))
    prices += [""] * (max_len - len(prices))

    for i in range(max_len):
        row = {
            "Purchase Order Number": final_op.get("Purchase Order Number", ""),
	    "Purchase Order Date": final_op.get("Docuemnt Invoice Date", ""),
            "Product": products[i],
            "Quantity": quantities[i],
            "Price per Kg": prices[i],
            "Dispatch Date": final_op.get("Dispatch date", ""),
            "Conditions": final_op.get("Conditions", ""),
            "Commission": "",
            "Dispatch Status": final_op.get("Dispatch Status", ""),
            "Airway Bill Date": final_op.get("Airway Bill Date", ""),
            "Commission Amount": "",
            "Invoice Due Date": final_op.get("Invoice Due Date", ""),
            "Party Invoice Number": final_op.get("Party Invoice Number", ""),
	    "Party Invoice Date": final_op.get("Party Invoice Date", ""),
            "Status of Debit Note": "",
        }
        flattened_rows.append(row)

    # 🧾 Convert to DataFrame and display
    table_df = pd.DataFrame(flattened_rows)

    st.markdown("### 🧾 Final Invoice Table View") 
    import streamlit.components.v1 as components

    # Convert DataFrame to TSV (tab-separated string)
    tsv_text = table_df.to_csv(index=False, sep='	')

    components.html(f"""
        <button onclick="copyTSV()">📋 Copy Table</button>
        <textarea id="tsvData" style="display:none;">{tsv_text}</textarea>
        <script>
        function copyTSV() {{
            const textArea = document.getElementById("tsvData");
            textArea.style.display = 'block';  // Make it visible temporarily
            textArea.select();
            document.execCommand('copy');
            textArea.style.display = 'none';   // Hide it again
            alert("✅ The Table is copied! Paste it in Excel/Sheets.");
        }}
        </script>
    """, height=50)

    st.markdown(f"#### 🏢 Company: {final_op.get('Company Name', '')}")
    st.dataframe(table_df, use_container_width=True)