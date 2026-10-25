import json
import random

from diagram_reader.synth import CLASS_IDS, CLASSES, GenerationConfig, generate_plan


def test_classes_are_unique_and_indexed():
    assert len({c.name for c in CLASSES}) == len(CLASSES)
    assert CLASS_IDS["arrow"] == 0 and CLASS_IDS["joint"] == len(CLASSES) - 1


def test_generate_plan_is_deterministic():
    cfg = GenerationConfig(width=1200, height=800, noise=False)
    _, t1 = generate_plan(cfg, random.Random(42))
    _, t2 = generate_plan(cfg, random.Random(42))
    assert t1.to_dict() == t2.to_dict()


def test_boxes_inside_image_and_yolo_normalised():
    cfg = GenerationConfig(width=1600, height=1000)
    for seed in range(5):
        img, truth = generate_plan(cfg, random.Random(seed))
        assert img.size == (1600, 1000)
        assert truth.objects, "un plan doit contenir des symboles"
        for a in truth.objects + truth.texts:
            x1, y1, x2, y2 = a.bbox
            assert 0 <= x1 < x2 <= 1600 and 0 <= y1 < y2 <= 1000, (a, seed)
        for line in truth.yolo_lines():
            cls, *vals = line.split()
            assert 0 <= int(cls) < len(CLASSES)
            assert all(0.0 < float(v) <= 1.0 for v in vals)


def test_every_signal_has_a_name_text_and_a_track():
    cfg = GenerationConfig()
    _, truth = generate_plan(cfg, random.Random(7))
    track_names = {t.name for t in truth.tracks}
    name_texts = {t.text for t in truth.texts if t.role == "signal_name"}
    for s in truth.signals:
        assert s.track in track_names
        assert s.name in name_texts
        assert s.pk_m >= truth.pk_origin_m


def test_truth_is_json_serialisable():
    _, truth = generate_plan(GenerationConfig(noise=False), random.Random(1))
    json.dumps(truth.to_dict())
