"""Convertit en lot des fichiers DWG en DXF.

Usage :
    python tools/dwg2dxf.py "C:/plans/dwg" "C:/plans/dxf"
    python tools/dwg2dxf.py "C:/plans/dwg" "C:/plans/dxf" --recursive --version ACAD2018

Deux moteurs, essayés dans cet ordre (ou forcés avec --engine) :

1. ODA File Converter (gratuit) : https://www.opendesign.com/guestfiles/oda_file_converter
   Détecté automatiquement dans "C:/Program Files/ODA/..." ou via --oda <chemin de l'exe>.
2. AutoCAD en mode console (accoreconsole.exe, livré avec AutoCAD) : détecté dans
   "C:/Program Files/Autodesk/AutoCAD 20xx/" ou via --accore <chemin de l'exe>.

Le script ne touche jamais aux DWG d'origine. Les DXF existants ne sont pas
reconvertis (sauf --force).
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DXF_VERSIONS = ("ACAD2000", "ACAD2004", "ACAD2007", "ACAD2010", "ACAD2013", "ACAD2018")


# --------------------------------------------------------------------- détection
def find_oda() -> Path | None:
    exe = shutil.which("ODAFileConverter")
    if exe:
        return Path(exe)
    patterns = [
        r"C:/Program Files/ODA/*/ODAFileConverter.exe",
        r"C:/Program Files (x86)/ODA/*/ODAFileConverter.exe",
        "/usr/bin/ODAFileConverter",
        "/opt/ODA/*/ODAFileConverter",
        "/Applications/ODAFileConverter.app/Contents/MacOS/ODAFileConverter",
    ]
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return Path(hits[-1])  # version la plus récente
    return None


def find_accore() -> Path | None:
    hits = sorted(glob.glob(r"C:/Program Files/Autodesk/AutoCAD *" "/accoreconsole.exe"))
    return Path(hits[-1]) if hits else None


# --------------------------------------------------------------------- moteurs
def convert_with_oda(oda: Path, src_dir: Path, dst_dir: Path, version: str, recursive: bool) -> int:
    """ODA convertit un dossier entier d'un coup : <exe> in out version DXF recursive audit [filtre]."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(oda), str(src_dir), str(dst_dir), version, "DXF", "1" if recursive else "0", "1", "*.DWG"]
    print("ODA :", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout, res.stderr, file=sys.stderr)
    return res.returncode


def convert_with_accore(accore: Path, dwg: Path, dxf: Path, version: str) -> bool:
    """AutoCAD console : un script SAVEAS par fichier."""
    ver_code = {"ACAD2000": "2000", "ACAD2004": "2004", "ACAD2007": "2007",
                "ACAD2010": "2010", "ACAD2013": "2013", "ACAD2018": "2018"}[version]
    dxf.parent.mkdir(parents=True, exist_ok=True)
    script = f'FILEDIA 0\nSAVEAS DXF 16 V{ver_code} "{dxf.as_posix()}"\nQUIT Y\n'
    with tempfile.NamedTemporaryFile("w", suffix=".scr", delete=False, encoding="ascii",
                                     errors="replace") as f:
        f.write(script)
        scr = f.name
    try:
        res = subprocess.run([str(accore), "/i", str(dwg), "/s", scr, "/l", "fr-FR"],
                             capture_output=True, text=True, timeout=300)
        ok = dxf.exists()
        if not ok:
            print(res.stdout[-800:], res.stderr[-800:], file=sys.stderr)
        return ok
    finally:
        os.unlink(scr)


# --------------------------------------------------------------------- principal
def main() -> int:
    p = argparse.ArgumentParser(description="Conversion DWG -> DXF en lot.")
    p.add_argument("src", type=Path, help="dossier contenant les .dwg")
    p.add_argument("dst", type=Path, help="dossier de sortie des .dxf")
    p.add_argument("--recursive", action="store_true", help="parcourir les sous-dossiers")
    p.add_argument("--version", default="ACAD2018", choices=DXF_VERSIONS, help="format DXF cible")
    p.add_argument("--engine", choices=["auto", "oda", "accore"], default="auto")
    p.add_argument("--oda", type=Path, help="chemin de ODAFileConverter.exe")
    p.add_argument("--accore", type=Path, help="chemin de accoreconsole.exe")
    p.add_argument("--force", action="store_true", help="reconvertir même si le DXF existe")
    args = p.parse_args()

    src, dst = args.src.resolve(), args.dst.resolve()
    if not src.is_dir():
        print(f"Dossier introuvable : {src}", file=sys.stderr)
        return 2
    pattern = "**/*.dwg" if args.recursive else "*.dwg"
    dwgs = sorted(f for f in src.glob(pattern) if f.suffix.lower() == ".dwg")
    if not dwgs:
        print(f"Aucun .dwg dans {src}")
        return 0
    print(f"{len(dwgs)} fichier(s) DWG trouvé(s)")

    oda = args.oda or (find_oda() if args.engine in ("auto", "oda") else None)
    accore = args.accore or (find_accore() if args.engine in ("auto", "accore") else None)

    # ---- ODA : conversion du dossier en une passe
    if oda and args.engine in ("auto", "oda"):
        if args.force:
            for d in dwgs:
                (dst / d.relative_to(src)).with_suffix(".dxf").unlink(missing_ok=True)
        rc = convert_with_oda(oda, src, dst, args.version, args.recursive)
        done = sum((dst / d.relative_to(src)).with_suffix(".dxf").exists() for d in dwgs)
        print(f"ODA : {done}/{len(dwgs)} DXF présents dans {dst}")
        if done == len(dwgs):
            return 0
        if rc == 0 and done:
            print("Certains fichiers n'ont pas été convertis (voir ci-dessus).")
            return 1
        print("ODA a échoué, essai avec AutoCAD…" if accore else "ODA a échoué.")

    # ---- AutoCAD console : fichier par fichier
    if accore and args.engine in ("auto", "accore"):
        ok = 0
        for d in dwgs:
            out = (dst / d.relative_to(src)).with_suffix(".dxf")
            if out.exists() and not args.force:
                ok += 1
                print(f"  = {d.name} (déjà converti)")
                continue
            print(f"  > {d.name}", end=" ", flush=True)
            if convert_with_accore(accore, d, out, args.version):
                ok += 1
                print("OK")
            else:
                print("ÉCHEC")
        print(f"AutoCAD : {ok}/{len(dwgs)} convertis dans {dst}")
        return 0 if ok == len(dwgs) else 1

    print(
        "Aucun convertisseur trouvé.\n"
        "  - Installez ODA File Converter (gratuit) : "
        "https://www.opendesign.com/guestfiles/oda_file_converter\n"
        "    puis relancez, ou passez --oda \"C:/Program Files/ODA/ODAFileConverter 26.x/ODAFileConverter.exe\"\n"
        "  - ou, avec AutoCAD installé, passez --accore "
        "\"C:/Program Files/Autodesk/AutoCAD 2025/accoreconsole.exe\"",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
