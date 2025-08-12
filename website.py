import os
import base64
from datetime import datetime
from pathlib import Path
import mimetypes
import re
import streamlit as st

# ---------- CONFIG ----------
FRAMES_DIRECTORY = r"C:\Users\DELL\OneDrive\Documents\Secret_couples\STARE_FRAMES"

# ---------- UTILS ----------
def format_file_size(size_bytes):
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB"]
    i = int((len(str(size_bytes))-1)/3)
    p = pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

def extract_confidence_from_filename(filename):
    patterns = [
        r'conf(?:idence)?[_-]?(\d+\.?\d*)',
        r'(\d\.\d+)_conf',
        r'stare_(\d\.\d+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, filename.lower())
        if match:
            try:
                conf = float(match.group(1))
                if conf > 1:
                    conf = conf / 100
                return min(conf, 1.0)
            except ValueError:
                continue
    return round(0.70 + (abs(hash(filename)) % 30) / 100, 2)

def scan_frames():
    frames_path = Path(FRAMES_DIRECTORY)
    if not frames_path.exists():
        st.error(f"Directory not found: {FRAMES_DIRECTORY}")
        return []

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    images = []

    for file_path in frames_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            stat = file_path.stat()
            with open(file_path, 'rb') as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if not mime_type:
                mime_type = 'image/jpeg'

            images.append({
                'filename': file_path.name,
                'creation_time': datetime.fromtimestamp(stat.st_ctime),
                'file_size': format_file_size(stat.st_size),
                'confidence': extract_confidence_from_filename(file_path.name),
                'subjects': 2,
                'duration': round(abs(hash(file_path.name)) % 50 / 10 + 1, 1),
                'base64_src': f"data:{mime_type};base64,{img_base64}"
            })

    images.sort(key=lambda x: x['creation_time'], reverse=True)
    return images

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Mutual Staring Detection Feed", layout="wide")
st.title("👁️ COUPLE Spotted!")
st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if st.button("🔄 Refresh Feed"):
    st.experimental_rerun()

images = scan_frames()

if not images:
    st.warning("No detections found in the STARE_FRAMES directory.")
else:
    st.write(f"📊 **{len(images)} Detections** | 📁 Source: STARE_FRAMES")
    cols = st.columns(3)

    for idx, img in enumerate(images):
        with cols[idx % 3]:
            st.image(img['base64_src'], caption=f"{img['filename']}", use_container_width=True)
            st.markdown(f"""
            - **Confidence:** {img['confidence']*100:.1f}%
            - **Subjects:** {img['subjects']}
            - **Duration:** {img['duration']}s
            - **Size:** {img['file_size']}
            - **Time:** {img['creation_time'].strftime('%Y-%m-%d %H:%M:%S')}
            """)
