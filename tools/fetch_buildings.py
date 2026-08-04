import json, urllib.parse, urllib.request, pathlib
BBOX=(47.3600,8.5100,47.3900,8.5600)
raw=pathlib.Path("buildings_raw.json")
if not raw.exists():
    s,w,n,e=BBOX
    q=f'[out:json][timeout:300];way["building"]({s},{w},{n},{e});out body geom;'
    req=urllib.request.Request("https://overpass.osm.ch/api/interpreter",data=urllib.parse.urlencode({"data":q}).encode())
    raw.write_bytes(urllib.request.urlopen(req,timeout=600).read())
d=json.loads(raw.read_text())
ways=[el for el in d["elements"] if el.get("type")=="way"]
print("building ways:",len(ways))
import collections
h=sum(1 for w in ways if "height" in w.get("tags",{}))
lv=sum(1 for w in ways if "building:levels" in w.get("tags",{}))
print(f"  with explicit height: {h}   with levels: {lv}   neither: {len(ways)-h-lv}")
print("  raw size %.1f MB"%(raw.stat().st_size/1e6))
