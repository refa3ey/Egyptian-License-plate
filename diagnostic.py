"""
DIAGNOSTIC TOOL
================
Checks your system health and identifies issues.

How to run:
  python diagnostic.py
"""

import os
import csv
import json
from collections import Counter


def check_files():
    print("="*60)
    print("FILE CHECK")
    print("="*60)
    
    files = {
        "labels.csv": "Training labels",
        "classes.json": "Class names",
        "model.pth1": "Trained model",
        "skipped.txt": "Skipped images"
    }
    
    for filename, desc in files.items():
        exists = "✅" if os.path.exists(filename) else "❌"
        print(f"{exists} {filename:20s} - {desc}")
    print()


def analyze_labels():
    print("="*60)
    print("LABEL DISTRIBUTION")
    print("="*60)
    
    if not os.path.exists("labels.csv"):
        print("❌ labels.csv not found")
        return
    
    labels = {}
    with open("labels.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) == 2:
                labels[row[0]] = row[1]
    
    counts = Counter(labels.values())
    total = len(labels)
    
    print(f"Total labeled: {total}")
    print(f"Total classes: {len(counts)}\n")
    
    weak_threshold = 30
    weak = {c: n for c, n in counts.items() if n < weak_threshold}
    
    if weak:
        print(f"⚠️  WEAK CLASSES (< {weak_threshold}):")
        for char, count in sorted(weak.items(), key=lambda x: x[1]):
            print(f"   {char:3s}: {count:3d} (need {weak_threshold-count} more)")
        print()
    
    print("ALL CLASSES:")
    for char, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * min(count // 5, 40)
        status = "⚠️ " if count < weak_threshold else "✅"
        print(f"{status} {char:3s}: {count:4d}  {bar}")
    print()
    
    print("RECOMMENDATIONS:")
    if weak:
        print(f"❌ {len(weak)} weak classes")
        print(f"   Label {weak_threshold}+ samples per class")
    else:
        print("✅ All classes sufficient")
    print()


def analyze_skipped():
    print("="*60)
    print("SKIPPED IMAGES")
    print("="*60)
    
    if not os.path.exists("skipped.txt"):
        print("No skipped.txt")
        return
    
    with open("skipped.txt", "r") as f:
        skipped = [line.strip() for line in f if line.strip()]
    
    print(f"Total skipped: {len(skipped)}")
    
    if len(skipped) > 0:
        patterns = Counter()
        for filename in skipped:
            if "darknoise" in filename:
                patterns["Dark + Noise"] += 1
            elif "noise" in filename:
                patterns["Noise"] += 1
            elif "blur" in filename:
                patterns["Blur"] += 1
            elif "bright14" in filename or "bright6" in filename:
                patterns["Extreme brightness"] += 1
            elif "rot-10" in filename or "rot10" in filename:
                patterns["Heavy rotation"] += 1
            else:
                patterns["Other"] += 1
        
        print("\nSkip reasons:")
        for pattern, count in patterns.most_common():
            print(f"  {pattern:20s}: {count:4d}")
    print()


def check_model_classes():
    print("="*60)
    print("MODEL vs LABELS CHECK")
    print("="*60)
    
    if not os.path.exists("classes.json"):
        print("❌ classes.json not found")
        return
    
    if not os.path.exists("labels.csv"):
        print("❌ labels.csv not found")
        return
    
    with open("classes.json", "r", encoding="utf-8") as f:
        model_classes = set(json.load(f))
    
    labels = {}
    with open("labels.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) == 2:
                labels[row[0]] = row[1]
    
    label_classes = set(labels.values())
    
    print(f"Model classes: {len(model_classes)}")
    print(f"Label classes: {len(label_classes)}\n")
    
    only_model = model_classes - label_classes
    only_labels = label_classes - model_classes
    
    if only_model:
        print("⚠️  In model but not labels:")
        print(f"   {sorted(only_model)}\n")
    
    if only_labels:
        print("⚠️  In labels but not model:")
        print(f"   {sorted(only_labels)}")
        print("   → Need to retrain!\n")
    
    if not only_model and not only_labels:
        print("✅ Model and labels synchronized")
    print()


def main():
    print("\n"+"="*60)
    print("  LICENSE PLATE DIAGNOSTIC TOOL")
    print("="*60+"\n")
    
    check_files()
    analyze_labels()
    analyze_skipped()
    check_model_classes()
    
    print("="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Review weak classes above")
    print("2. Label more: python labeling.py")
    print("3. Train: python train.py")
    print("4. Test: python predict.py <image>")
    print()


if __name__ == "__main__":
    main()