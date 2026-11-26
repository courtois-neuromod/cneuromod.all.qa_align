"""Write the dashboard: one HTML file with its data inside, no network needed.

`template.html` (next to this file) holds the page; the data of
`analysis.dashboard.payload` replaces the placeholder in its
`<script id="qa-data" type="application/json">` element.
"""

import json
from pathlib import Path

TEMPLATE = Path(__file__).with_name("template.html")
PLACEHOLDER = "__QA_ALIGN_DATA__"


def render(data, template_path=TEMPLATE):
    """The page's HTML with `data` embedded."""
    template = Path(template_path).read_text()
    if PLACEHOLDER not in template:
        raise ValueError(f"{template_path} has no {PLACEHOLDER} placeholder")
    encoded = json.dumps(data, separators=(",", ":"), allow_nan=False)
    # "</" must not appear inside a <script> element, whatever the data holds
    return template.replace(PLACEHOLDER, encoded.replace("</", "<\\/"))


def write_page(data, output_path, template_path=TEMPLATE):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render(data, template_path))
    return output_path
