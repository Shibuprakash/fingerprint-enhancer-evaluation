"""
Fingerprint verification pipeline with biometric evaluation (FAR / FRR / EER).

Pipeline: load -> (optional) Gabor enhancement -> ORB keypoints -> matching -> score
Experiment: compare accuracy WITH vs WITHOUT enhancement.

Dataset: FVC2002 DB1_B (80 images, 10 fingers x 8 impressions).
Filenames look like 101_1.tif  ->  finger id = 101, impression = 1

Usage:
    python fingerprint_eval.py --data path/to/DB1_B
    python fingerprint_eval.py --data path/to/DB1_B --no-enhance
"""
import argparse, glob, itertools, os, time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def load_images(folder):
    paths = sorted(glob.glob(os.path.join(folder, "*.tif")) +
                   glob.glob(os.path.join(folder, "*.png")) +
                   glob.glob(os.path.join(folder, "*.bmp")))
    imgs, ids = [], []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        imgs.append(img)
        ids.append(os.path.basename(p).split("_")[0])  # finger id
    return imgs, ids


def preprocess(img, enhance):
    if not enhance:
        return cv2.equalizeHist(img)  # baseline: simple contrast normalisation
    import fingerprint_enhancer
    fn = getattr(fingerprint_enhancer, "enhance_fingerprint", None) or \
         getattr(fingerprint_enhancer, "enhance_Fingerprint")  # name differs across versions
    try:
        out = fn(img)  # oriented Gabor filtering
    except Exception:
        return None  # enhancement failed (e.g. very poor quality print)
    return (np.asarray(out) > 0).astype(np.uint8) * 255


def extract(img, orb):
    kp, des = orb.detectAndCompute(img, None)
    return kp, des


def match_score(des1, des2, bf):
    """Score = fraction of descriptors that pass Lowe's ratio test."""
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return 0.0
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < 0.75 * n.distance]
    return len(good) / min(len(des1), len(des2))


def compute_eer(genuine, impostor):
    y = np.r_[np.ones(len(genuine)), np.zeros(len(impostor))]
    s = np.r_[genuine, impostor]
    far, tpr, thr = roc_curve(y, s)  # FAR = false positive rate
    frr = 1 - tpr                    # FRR = false negative rate
    i = np.nanargmin(np.abs(far - frr))
    return (far[i] + frr[i]) / 2, thr[i], far, frr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--no-enhance", action="store_true")
    args = ap.parse_args()
    enhance = not args.no_enhance
    tag = "enhanced" if enhance else "baseline"

    imgs, ids = load_images(args.data)
    print(f"Loaded {len(imgs)} images, {len(set(ids))} fingers  [{tag}]")

    orb = cv2.ORB_create(nfeatures=1000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    t0 = time.time()
    feats, fte = [], 0
    for im in imgs:
        pre = preprocess(im, enhance)
        if pre is None:  # count as Failure To Enrol, fall back to baseline
            fte += 1
            pre = preprocess(im, False)
        feats.append(extract(pre, orb)[1])
    if fte:
        print(f"Failure-to-enrol (enhancement failed): {fte}/{len(imgs)} images")
    print(f"Feature extraction: {(time.time() - t0) / len(imgs) * 1000:.1f} ms/image")

    genuine, impostor = [], []
    for i, j in itertools.combinations(range(len(imgs)), 2):
        s = match_score(feats[i], feats[j], bf)
        (genuine if ids[i] == ids[j] else impostor).append(s)

    eer, thr, far, frr = compute_eer(np.array(genuine), np.array(impostor))
    print(f"Genuine pairs: {len(genuine)}, impostor pairs: {len(impostor)}")
    print(f"EER = {eer * 100:.2f}%  at threshold {thr:.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(impostor, bins=40, alpha=0.6, density=True, label="Impostor")
    ax[0].hist(genuine, bins=40, alpha=0.6, density=True, label="Genuine")
    ax[0].axvline(thr, color="k", ls="--", label="EER threshold")
    ax[0].set(title=f"Score distribution ({tag})", xlabel="Match score")
    ax[0].legend()
    ax[1].plot(far * 100, frr * 100)
    ax[1].plot([0, 100], [0, 100], "k:", lw=0.8)
    ax[1].set(title=f"FAR vs FRR  (EER = {eer * 100:.2f}%)", xlabel="FAR (%)", ylabel="FRR (%)",
              xscale="log", xlim=(0.1, 100))
    plt.tight_layout()
    plt.savefig(f"results_{tag}.png", dpi=120)
    print(f"Saved results_{tag}.png")


if __name__ == "__main__":
    main()
