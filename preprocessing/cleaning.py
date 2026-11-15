import re, html

def clean_text(t):
    t = html.unescape(str(t))
    t = t.encode('ascii', 'ignore').decode()
    t = t.lower()
    t = re.sub(r'<.*?>', ' ', t)
    t = re.sub(r'http\S+|www.\S+', ' url ', t)
    t = re.sub(r'\S+@\S+', ' email ', t)
    t = re.sub(r'\d+', ' num ', t)
    t = re.sub(r'[^a-z\s]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()
