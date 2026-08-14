"""Label Studio D1 setup — idempotent: project + config + images + tasks.

Hard-won lessons (2026-08-15, Python 3.14 + LS 1.23.0 on Windows):
1. pip 1.23 wheel ships the SPA under top-level `web/` which pip DROPS
   (non-package dir). Fix: unzip it into site-packages manually:
     python -c "import zipfile,glob; z=zipfile.ZipFile(glob.glob('label_studio*.whl')[0]); [z.extract(n, r'D:\\programs\\Python\\Lib\\site-packages') for n in z.namelist() if n.startswith('web/')]"
2. label_config MUST use the object+labels split form (`<Polygon>` +
   separate `<Labels toName="poly">`). The combined `<PolygonLabels>` tag
   parses but registers NO toolbar tool (only move/pan/zoom show).
3. Images are served via FileUpload DB records; manual copies to media/upload
   ills 500. Files passed to /api/projects/{id}/import register records.
4. Legacy API tokens are disabled; use session login (SessionAuthentication).

Usage:
    python tools/labelstudio_setup.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import requests

BASE = "http://localhost:8080"
EMAIL = "admin@localhost"
PASSWORD = "crux-d1-admin"

LABEL_CONFIG = """<View>
<Image name="image" value="$image"/>
<Polygon name="poly" toName="image"/>
<Labels name="label" toName="poly">
<Label value="hold" background="#4AA8FF"/>
<Label value="volume" background="#FF6675"/>
</Labels>
</View>"""


def login() -> requests.Session:
    s = requests.Session()
    s.get(f"{BASE}/user/login")
    csrf = s.cookies.get("csrftoken", "")
    r = s.post(f"{BASE}/user/login",
               data={"email": EMAIL, "password": PASSWORD,
                     "csrfmiddlewaretoken": csrf},
               headers={"Referer": f"{BASE}/user/login"})
    assert r.status_code == 200, f"login failed {r.status_code}"
    return s


def setup(project_title: str, coco_path: Path, photos_dir: Path) -> None:
    s = login()
    # find or create project
    proj = None
    for p in s.get(f"{BASE}/api/projects/").json():
        if p["title"] == project_title:
            proj = p
            break
    if proj is None:
        r = s.post(f"{BASE}/api/projects/",
                   json={"title": project_title,
                         "description": "D1 pre-annotation correction",
                         "label_config": LABEL_CONFIG})
        assert r.status_code in (200, 201), r.text[:200]
        proj = r.json()
    pid = proj["id"]
    print(f"project {pid}: {proj['title']}")

    # (re)apply config (idempotent, validated)
    r = s.patch(f"{BASE}/api/projects/{pid}/", json={"label_config": LABEL_CONFIG})
    assert r.status_code == 200, r.text[:200]

    # import images + task json in ONE call so FileUpload records exist
    files = [("file", (f.name, open(f, "rb"), "image/jpeg"))
             for f in sorted(photos_dir.glob("*.jpg"))]
    files.append(("file", ("tasks.json", open(coco_path, "rb"), "application/json")))
    r = s.post(f"{BASE}/api/projects/{pid}/import", files=files)
    print("import:", r.status_code, r.text[:150])
    print(f"open {BASE}/projects/{pid}/data -> login -> annotate")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default="CRUX D1 realpic 标注")
    ap.add_argument("--coco", type=Path,
                    default=Path("data/preannotations/labelstudio_tasks.json"))
    ap.add_argument("--photos", type=Path, default=Path("data/realpic"))
    args = ap.parse_args()
    setup(args.title, args.coco, args.photos)
