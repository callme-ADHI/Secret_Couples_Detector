import os
import base64
import json
from datetime import datetime
from pathlib import Path
import mimetypes

class StareFrameWebsiteGenerator:
    def __init__(self, frames_directory, output_file="stare_detection_feed.html"):
        self.frames_directory = frames_directory
        self.output_file = output_file
        self.image_data = []
    
    def scan_frames_directory(self):
        """Scan the STARE_FRAMES directory and collect all images with metadata"""
        frames_path = Path(self.frames_directory)
        
        if not frames_path.exists():
            print(f"Directory not found: {self.frames_directory}")
            return []
        
        # Supported image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
        
        images = []
        for file_path in frames_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                try:
                    # Get file stats
                    stat = file_path.stat()
                    creation_time = datetime.fromtimestamp(stat.st_ctime)
                    modification_time = datetime.fromtimestamp(stat.st_mtime)
                    file_size = stat.st_size
                    
                    # Convert image to base64
                    with open(file_path, 'rb') as img_file:
                        img_data = img_file.read()
                        img_base64 = base64.b64encode(img_data).decode('utf-8')
                        
                    # Get MIME type
                    mime_type, _ = mimetypes.guess_type(str(file_path))
                    if not mime_type:
                        mime_type = 'image/jpeg'
                    
                    # Extract confidence from filename if present
                    confidence = self.extract_confidence_from_filename(file_path.name)
                    
                    image_info = {
                        'id': len(images) + 1,
                        'filename': file_path.name,
                        'filepath': str(file_path),
                        'creation_time': creation_time,
                        'modification_time': modification_time,
                        'file_size': file_size,
                        'base64_data': img_base64,
                        'mime_type': mime_type,
                        'confidence': confidence,
                        'subjects': 2,  # Default, you can modify this based on your detection
                        'duration': round(abs(hash(file_path.name)) % 50 / 10 + 1, 1)  # Mock duration
                    }
                    
                    images.append(image_info)
                    print(f"Processed: {file_path.name}")
                    
                except Exception as e:
                    print(f"Error processing {file_path.name}: {str(e)}")
                    continue
        
        # Sort by creation time (newest first)
        images.sort(key=lambda x: x['creation_time'], reverse=True)
        self.image_data = images
        print(f"Found and processed {len(images)} images")
        return images
    
    def extract_confidence_from_filename(self, filename):
        """Try to extract confidence score from filename"""
        # Look for patterns like conf_0.85 or confidence_85 etc.
        import re
        
        # Pattern for confidence in filename
        confidence_patterns = [
            r'conf(?:idence)?[_-]?(\d+\.?\d*)',
            r'(\d\.\d+)_conf',
            r'stare_(\d\.\d+)',
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, filename.lower())
            if match:
                try:
                    conf = float(match.group(1))
                    # If it's a percentage (>1), convert to decimal
                    if conf > 1:
                        conf = conf / 100
                    return min(conf, 1.0)  # Cap at 1.0
                except ValueError:
                    continue
        
        # Default confidence if not found in filename
        return round(0.70 + (abs(hash(filename)) % 30) / 100, 2)
    
    def format_file_size(self, size_bytes):
        """Convert bytes to human readable format"""
        if size_bytes == 0:
            return "0 B"
        size_names = ["B", "KB", "MB", "GB"]
        import math
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"
    
    def generate_html(self):
        """Generate the complete HTML file with embedded images"""
        
        if not self.image_data:
            print("No images found to generate website")
            return
        
        # JavaScript data for the images
        js_image_data = []
        for img in self.image_data:
            js_image_data.append({
                'id': img['id'],
                'filename': img['filename'],
                'timestamp': img['creation_time'].strftime('%Y-%m-%d %H:%M:%S'),
                'confidence': img['confidence'],
                'subjects': img['subjects'],
                'duration': img['duration'],
                'file_size': self.format_file_size(img['file_size']),
                'base64_src': f"data:{img['mime_type']};base64,{img['base64_data']}"
            })
        
        html_template = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mutual Staring Detection Feed</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 30px;
            text-align: center;
            position: relative;
        }}

        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        }}

        .header p {{
            font-size: 1.1rem;
            opacity: 0.9;
        }}

        .status-indicator {{
            position: absolute;
            top: 20px;
            right: 30px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .status-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #4CAF50;
            animation: pulse 2s infinite;
        }}

        .status-text {{
            font-size: 0.9rem;
            opacity: 0.9;
        }}

        @keyframes pulse {{
            0% {{ transform: scale(1); opacity: 1; }}
            50% {{ transform: scale(1.2); opacity: 0.7; }}
            100% {{ transform: scale(1); opacity: 1; }}
        }}

        .controls {{
            padding: 20px 40px;
            border-bottom: 1px solid #eee;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 15px;
        }}

        .control-group {{
            display: flex;
            gap: 15px;
            align-items: center;
        }}

        .control-btn {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 0.9rem;
        }}

        .control-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }}

        .stats {{
            display: flex;
            gap: 20px;
            font-size: 0.9rem;
            color: #666;
        }}

        .stat-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}

        .main-content {{
            padding: 40px;
            max-height: 70vh;
            overflow-y: auto;
        }}

        .feed {{
            display: flex;
            flex-direction: column;
            gap: 25px;
        }}

        .detection-post {{
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
            border-left: 5px solid #4facfe;
        }}

        .detection-post:hover {{
            transform: translateY(-3px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
        }}

        .post-header {{
            padding: 20px 25px 15px;
            background: linear-gradient(135deg, #f8f9ff 0%, #f0f4ff 100%);
            border-bottom: 1px solid #eee;
        }}

        .post-title {{
            font-size: 1.2rem;
            font-weight: 600;
            color: #333;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }}

        .detection-badge {{
            background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
            color: white;
            padding: 4px 12px;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: 500;
        }}

        .post-meta {{
            font-size: 0.9rem;
            color: #666;
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}

        .post-image {{
            position: relative;
            overflow: hidden;
        }}

        .post-image img {{
            width: 100%;
            height: 400px;
            object-fit: cover;
            transition: transform 0.3s ease;
            cursor: pointer;
        }}

        .post-image:hover img {{
            transform: scale(1.02);
        }}

        .image-overlay {{
            position: absolute;
            top: 15px;
            left: 15px;
            background: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 8px 15px;
            border-radius: 20px;
            font-size: 0.85rem;
            backdrop-filter: blur(10px);
        }}

        .post-actions {{
            padding: 20px 25px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .action-buttons {{
            display: flex;
            gap: 10px;
        }}

        .action-btn {{
            background: transparent;
            border: 2px solid #ddd;
            color: #666;
            padding: 8px 15px;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 0.85rem;
        }}

        .action-btn:hover {{
            background: #4facfe;
            color: white;
            border-color: #4facfe;
        }}

        .timestamp {{
            font-size: 0.8rem;
            color: #999;
        }}

        .empty-feed {{
            text-align: center;
            padding: 80px 20px;
            color: #666;
        }}

        .empty-icon {{
            font-size: 5rem;
            margin-bottom: 20px;
            opacity: 0.5;
        }}

        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.9);
        }}

        .modal-content {{
            position: relative;
            margin: auto;
            display: block;
            width: 90%;
            max-width: 800px;
            max-height: 90%;
            object-fit: contain;
            top: 50%;
            transform: translateY(-50%);
            border-radius: 10px;
        }}

        .close {{
            position: absolute;
            top: 20px;
            right: 35px;
            color: #fff;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }}

        @media (max-width: 768px) {{
            .main-content {{
                padding: 20px;
            }}
            .post-image img {{
                height: 250px;
            }}
            .post-meta {{
                gap: 10px;
            }}
            .post-title {{
                font-size: 1rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="status-indicator">
                <div class="status-dot"></div>
                <span class="status-text">Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
            </div>
            <h1>👁️ Mutual Staring Detection Feed</h1>
            <p>Couple Spoted !!!</p>
        </div>

        <div class="controls">
            <div class="control-group">
                <button class="control-btn" onclick="location.reload()">🔄 Refresh</button>
                <button class="control-btn" onclick="toggleFullscreen()">🖥️ Fullscreen</button>
            </div>
            <div class="stats">
                <div class="stat-item">
                    <span>📊</span>
                    <span>{len(self.image_data)} Detection{'s' if len(self.image_data) != 1 else ''}</span>
                </div>
                <div class="stat-item">
                    <span>📁</span>
                    <span>STARE_FRAMES</span>
                </div>
            </div>
        </div>

        <div class="main-content">
            <div class="feed" id="feed"></div>
        </div>
    </div>

    <div id="imageModal" class="modal">
        <span class="close" onclick="closeModal()">&times;</span>
        <img class="modal-content" id="modalImage">
    </div>

    <script>
        const detectionData = {json.dumps(js_image_data, indent=8)};

        function renderFeed() {{
            const feed = document.getElementById('feed');
            
            if (detectionData.length === 0) {{
                feed.innerHTML = `
                    <div class="empty-feed">
                        <div class="empty-icon">👁️‍🗨️</div>
                        <h3>No detections found</h3>
                        <p>No images found in the STARE_FRAMES directory</p>
                    </div>
                `;
                return;
            }}

            feed.innerHTML = detectionData.map(detection => `
                <div class="detection-post">
                    <div class="post-header">
                        <div class="post-title">
                            🎯 Mutual Staring Detected!
                            <span class="detection-badge">Confidence: ${{(detection.confidence * 100).toFixed(1)}}%</span>
                        </div>
                        <div class="post-meta">
                            <span>📁 ${{detection.filename}}</span>
                            <span>👥 ${{detection.subjects}} subjects</span>
                            <span>⏱️ ${{detection.duration}}s</span>
                            <span>📏 ${{detection.file_size}}</span>
                        </div>
                    </div>
                    <div class="post-image">
                        <div class="image-overlay">Detected Frame</div>
                        <img src="${{detection.base64_src}}" 
                             alt="Detection ${{detection.id}}" 
                             onclick="openModal(this.src)">
                    </div>
                    <div class="post-actions">
                        <div class="action-buttons">
                            <button class="action-btn" onclick="downloadImage('${{detection.filename}}', '${{detection.base64_src}}')">
                                💾 Download
                            </button>
                            <button class="action-btn" onclick="copyToClipboard('${{detection.filename}}')">
                                📋 Copy Name
                            </button>
                            <button class="action-btn" onclick="showImageInfo(${{detection.id}})">
                                ℹ️ Info
                            </button>
                        </div>
                        <div class="timestamp">${{detection.timestamp}}</div>
                    </div>
                </div>
            `).join('');
        }}

        function openModal(src) {{
            const modal = document.getElementById('imageModal');
            const modalImg = document.getElementById('modalImage');
            modal.style.display = 'block';
            modalImg.src = src;
        }}

        function closeModal() {{
            document.getElementById('imageModal').style.display = 'none';
        }}

        function downloadImage(filename, base64Src) {{
            const link = document.createElement('a');
            link.download = filename;
            link.href = base64Src;
            link.click();
        }}

        function copyToClipboard(text) {{
            navigator.clipboard.writeText(text).then(() => {{
                alert('Filename copied to clipboard: ' + text);
            }});
        }}

        function showImageInfo(id) {{
            const detection = detectionData.find(d => d.id === id);
            if (detection) {{
                alert(`Image Information:\\n\\nFilename: ${{detection.filename}}\\nTimestamp: ${{detection.timestamp}}\\nConfidence: ${{(detection.confidence * 100).toFixed(1)}}%\\nSubjects: ${{detection.subjects}}\\nDuration: ${{detection.duration}}s\\nFile Size: ${{detection.file_size}}`);
            }}
        }}

        function toggleFullscreen() {{
            if (!document.fullscreenElement) {{
                document.documentElement.requestFullscreen();
            }} else {{
                document.exitFullscreen();
            }}
        }}

        // Close modal when clicking outside
        window.onclick = function(event) {{
            const modal = document.getElementById('imageModal');
            if (event.target === modal) {{
                closeModal();
            }}
        }}

        // Initialize
        renderFeed();
    </script>
</body>
</html>'''
        
        # Write the HTML file
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                f.write(html_template)
            print(f"Website generated successfully: {self.output_file}")
            print(f"Total images embedded: {len(self.image_data)}")
            
            # Open the file in default browser
            import webbrowser
            file_path = os.path.abspath(self.output_file)
            webbrowser.open(f'file://{file_path}')
            print(f"Website opened in browser: {file_path}")
            
        except Exception as e:
            print(f"Error writing HTML file: {str(e)}")

def main():
    # Configuration
    FRAMES_DIRECTORY = r"C:\\Users\\DELL\\OneDrive\\Documents\\Secret_couples\\STARE_FRAMES"
    OUTPUT_FILE = "stare_detection_feed.html"
    
    print("🔍 Mutual Staring Detection Website Generator")
    print("=" * 50)
    print(f"Scanning directory: {FRAMES_DIRECTORY}")
    
    # Create generator instance
    generator = StareFrameWebsiteGenerator(FRAMES_DIRECTORY, OUTPUT_FILE)
    
    # Scan for images
    images = generator.scan_frames_directory()
    
    if images:
        print(f"\\n📊 Found {len(images)} images")
        print(f"📅 Date range: {images[-1]['creation_time'].strftime('%Y-%m-%d %H:%M')} to {images[0]['creation_time'].strftime('%Y-%m-%d %H:%M')}")
        
        # Generate website
        print("\\n🌐 Generating website...")
        generator.generate_html()
        print("\\n✅ Website generation complete!")
    else:
        print("\\n❌ No images found in the specified directory")
        print("Make sure your detection script is saving images to:")
        print(f"   {FRAMES_DIRECTORY}")

if __name__ == "__main__":
    main()