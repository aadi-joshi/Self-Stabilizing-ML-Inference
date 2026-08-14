import os
import base64

HERE = os.path.dirname(__file__)
key = None
with open(os.path.join(HERE, '..', '.env')) as f:
    for line in f:
        if line.startswith('OPENAI_API_KEY'):
            key = line.strip().split('=', 1)[1]

from openai import OpenAI
client = OpenAI(api_key=key)

FIGDIR = os.path.join(HERE, 'figures')
os.makedirs(FIGDIR, exist_ok=True)

PROMPT_GRAPHICAL_ABSTRACT = """
A clean, minimalist scientific graphical-abstract illustration for a machine learning research paper, flat vector style, white background, restrained color palette (navy blue, slate gray, one warm orange accent color), no text or labels anywhere in the image.

Left side: five small distinct abstract icons in a vertical row representing different neural network architecture families (a grid pattern for a convolutional network, a chain of connected blocks for a residual network, a simple layered rectangle for a plain multilayer perceptron, a grid of small squares with an eye/attention symbol for a transformer, a checkerboard pattern for a mixer network) - each icon a different color to show they are distinct.

Center: each of the five icons connects via a thin line to a single central dial or knob mechanism, like a control valve or gauge, rendered in orange, that all five lines pass through before continuing to the right.

Right side: a single output gauge or threshold meter, showing the lines converging to the same reading, implying one shared output.

The overall composition should visually communicate: many different things feeding through one shared control mechanism to produce one shared outcome. Professional, elegant, suitable for a journal figure, isometric or flat 2D technical illustration style, no photorealism, no 3D render, no text, no numbers, no words anywhere in the image.
"""

resp = client.images.generate(
    model="gpt-image-1",
    prompt=PROMPT_GRAPHICAL_ABSTRACT,
    size="1536x1024",
    n=1,
)
img_b64 = resp.data[0].b64_json
with open(os.path.join(FIGDIR, 'graphical_abstract_raw.png'), 'wb') as f:
    f.write(base64.b64decode(img_b64))
print("wrote graphical_abstract_raw.png")
