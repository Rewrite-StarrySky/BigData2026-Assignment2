"""Rasterise an SVG with headless Chrome at its natural size (used for the slides)."""
import re, subprocess, sys
from pathlib import Path
def svg_to_png(svg, png, scale=2):
    s = Path(svg).read_text(); w = float(re.search(r'width="([\d.]+)pt"', s).group(1)) * 96 / 72; h = float(re.search(r'height="([\d.]+)pt"', s).group(1)) * 96 / 72
    html = Path(svg).with_suffix('.tmp.html'); html.write_text(f'<html><body style="margin:0;background:white"><img src="{Path(svg).name}" width="{w}" height="{h}"></body></html>')
    subprocess.run(['google-chrome', '--headless=new', '--no-sandbox', '--disable-gpu', '--hide-scrollbars', f'--force-device-scale-factor={scale}', f'--window-size={int(w) + 2},{int(h) + 2}', f'--screenshot={Path(png).resolve()}', html.resolve().as_uri()], capture_output=True, check=True); html.unlink()
    return int(w), int(h)
if __name__ == '__main__':
    for f in sys.argv[1:]: print(f, svg_to_png(f, f.replace('.svg', '.png')))
