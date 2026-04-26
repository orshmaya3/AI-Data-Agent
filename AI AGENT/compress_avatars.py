from PIL import Image
import os

AVATARS_DIR = os.path.join(os.path.dirname(__file__), 'flask_static', 'avatars')
for fname in os.listdir(AVATARS_DIR):
    if fname.endswith('.png'):
        path = os.path.join(AVATARS_DIR, fname)
        before = os.path.getsize(path)
        img = Image.open(path).convert('RGBA')
        img = img.resize((200, 200), Image.LANCZOS)
        img.save(path, 'PNG', optimize=True)
        after = os.path.getsize(path)
        print(f"{fname}: {before//1024}KB -> {after//1024}KB")
