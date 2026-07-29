import json

from detect_objects import detect_objects
from color_palette import extract_colors
from lighting_score import analyze_lighting
from generate_tags import generate_tags

image_path = "images/sample.jpg"

objects = detect_objects(image_path)

colors = extract_colors(image_path)

brightness, contrast, lighting = analyze_lighting(image_path)

tags = generate_tags(objects, colors, lighting)

metadata = {
    "Objects": objects,
    "Dominant Colors": colors,
    "Brightness": round(brightness,2),
    "Contrast": round(contrast,2),
    "Lighting": lighting,
    "Tags": tags
}

print(metadata)

with open("output/result.json","w") as f:
    json.dump(metadata,f,indent=4)