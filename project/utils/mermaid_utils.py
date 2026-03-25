import re
import PIL
import base64
import requests
from io import BytesIO

from reportlab import platypus
from reportlab.lib.units import cm
from reportlab.lib.colors import blue
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle


def mm(graph):
    graphbytes = graph.encode("utf8")
    base64_bytes = base64.b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")

    return "https://mermaid.ink/img/" + base64_string


def mms(graph):
    graphbytes = graph.encode("utf8")
    base64_bytes = base64.b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")

    return "https://mermaid.ink/svg/" + base64_string


def convert_mm(content):
    pattern = r'```(.*?)```'
    matches = re.findall(pattern, content,re.DOTALL)
    modified_strings = [match.replace('\n', ';') for match in matches]
    
    for original, modified in zip(matches, modified_strings):
        content = content.replace(f'```{original}```', f'{modified}')    
        
    return content


def image_mm(line,L):
    line = line.replace("mermaid;","")

    url = mm(line)
    url_svg = mms(line)

    try:
        res = requests.get(url, timeout=10)
        x = PIL.Image.open(BytesIO(res.content))
        imgdata = BytesIO()
        x.save(imgdata,'JPEG')
        imgdata.seek(0)

        if x.height > x.width:
            ratio = 12.16/x.height
            xd = x.width * ratio
            L.append(platypus.Image(imgdata,xd * cm, 12.16 * cm))

        else:
            ratio = 16/x.width
            xd = x.height * ratio
            L.append(platypus.Image(imgdata,16 * cm, xd * cm))

        L.append(Paragraph(f"<a href={url_svg}>show svg</a>",ParagraphStyle(name='fd',fontName='맑은고딕',fontSize=12,leading=20, textColor=blue)))
    except Exception:
        pass