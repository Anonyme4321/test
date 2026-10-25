"""Ligne de commande : `rdr-synth --out data/synth -n 50 --seed 0`.

Produit une arborescence prête pour Ultralytics :

    out/
      images/{train,val}/plan_000000.png
      labels/{train,val}/plan_000000.txt      (YOLO : cls cx cy w h normalisés)
      meta/plan_000000.json                   (vérité terrain complète)
      data.yaml
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import ImageDraw

from diagram_reader.synth.classes import CLASS_NAMES
from diagram_reader.synth.generator import GenerationConfig, PlanTruth, generate_plan


def draw_preview(img, truth: PlanTruth):
    """Dessine les boîtes (symboles en rouge, textes en bleu) pour contrôle visuel."""
    out = img.convert("RGB")
    d = ImageDraw.Draw(out)
    for a in truth.objects:
        d.rectangle(a.bbox, outline=(220, 30, 30), width=2)
        d.text((a.bbox[0], a.bbox[1] - 12), a.cls, fill=(220, 30, 30))
    for t in truth.texts:
        d.rectangle(t.bbox, outline=(30, 90, 220), width=1)
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Génère des plans de signalisation synthétiques annotés.")
    p.add_argument("--out", type=Path, default=Path("data/synth"))
    p.add_argument("-n", "--count", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--width", type=int, default=2400)
    p.add_argument("--height", type=int, default=1600)
    p.add_argument("--no-noise", action="store_true")
    p.add_argument("--preview", action="store_true", help="écrit aussi preview/*.jpg avec les boîtes")
    args = p.parse_args(argv)

    rng = random.Random(args.seed)
    cfg = GenerationConfig(width=args.width, height=args.height, noise=not args.no_noise)
    for sub in ("images/train", "images/val", "labels/train", "labels/val", "meta"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    if args.preview:
        (args.out / "preview").mkdir(exist_ok=True)

    n_val = round(args.count * args.val_ratio)
    for i in range(args.count):
        split = "val" if i < n_val else "train"
        name = f"plan_{i:06d}"
        img, truth = generate_plan(cfg, rng)
        img.save(args.out / "images" / split / f"{name}.png")
        (args.out / "labels" / split / f"{name}.txt").write_text("\n".join(truth.yolo_lines()) + "\n")
        (args.out / "meta" / f"{name}.json").write_text(
            json.dumps(truth.to_dict(), ensure_ascii=False, indent=1)
        )
        if args.preview:
            draw_preview(img, truth).save(args.out / "preview" / f"{name}.jpg", quality=85)
        if (i + 1) % 10 == 0 or i + 1 == args.count:
            print(f"{i + 1}/{args.count}")

    data_yaml = (
        f"path: {args.out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {list(CLASS_NAMES)}\n"
    )
    (args.out / "data.yaml").write_text(data_yaml)
    print(f"OK -> {args.out}  ({args.count} plans, {n_val} en validation)")


if __name__ == "__main__":
    main()
