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
A sophisticated scientific graphical abstract for a machine learning research paper, in the style of a Nature Physics or Science journal editorial illustration: fine engraved technical linework, subtle depth and shading, muted precise palette (deep navy, warm brass/copper, warm off-white paper background), the rendering quality of a professional scientific illustrator, NOT flat corporate icon art, NOT flaticon-style clipart, NOT a business infographic.

Composition, left to right:

Left: five distinct neural network architecture diagrams stacked vertically, each rendered with real technical structure, not generic icons: (1) a convolutional network shown as a small stack of layered feature-map cubes with a sliding kernel window; (2) a residual network shown as a chain of blocks with curved skip-connection arcs looping over them; (3) a plain multilayer perceptron shown as columns of small circular nodes fully connected by fine lines; (4) a vision transformer shown as a grid of image patches with a few attention lines radiating from a central patch to others; (5) an MLP-mixer shown as a grid with alternating cross-hatched rows and columns representing token-mixing and channel-mixing. Each diagram is small, precise, monochrome navy linework on the paper background, no color fill except thin brass accent lines. No text, letters, or labels anywhere near these five diagrams.

Center: all five architectures connect via fine converging brass lines into a single antique mechanical governor mechanism, like a Watt's centrifugal flyball governor from classical engineering, rendered in warm brass/copper with visible mechanical linkages, spinning arms, and a central spindle, symbolizing a shared feedback control mechanism that regulates all five inputs identically.

Right: the governor's output connects to exactly one single clean analog dial gauge with a needle pointing to one shared position, rendered in navy with a brass needle. There must be only one gauge and one governor mechanism in the entire image; the bottom-right corner and all other empty space in the image should remain blank paper background, with no second smaller device, no duplicate valve, no extra mechanism anywhere.

The whole piece should look like it belongs on the cover of a physics or machine-learning journal: elegant, precise, slightly vintage-technical in feeling, with real engineering detail in the mechanism, not a smooth flat vector icon set. No text, no numbers, no words, no labels anywhere in the image.
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
