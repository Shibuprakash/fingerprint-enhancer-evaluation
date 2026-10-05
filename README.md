# Fingerprint Verification with Biometric Evaluation

Fingerprint verification pipeline on FVC2002 DB1_B with an ablation study
on Gabor enhancement, evaluated using standard biometric metrics.

## Pipeline
1. Preprocessing: histogram equalization (baseline) or oriented Gabor enhancement
2. Feature extraction: ORB keypoints and binary descriptors
3. Matching: Hamming distance with Lowe's ratio test, normalized score
4. Evaluation: all-pairs comparison (280 genuine, 2,880 impostor), FAR / FRR / EER

## Results
| Mode | EER | FRR at FAR = 1% | FRR at FAR = 0.1% | Time (ms/image) |
|------|-----|-----------------|-------------------|-----------------|
| Baseline (histogram equalization) | **28.88%** | ~74% | ~80% | (fill in) |
| Gabor enhanced | **30.35%** | ~80% | ~88% | (fill in) |



## Usage
pip install fingerprint_enhancer opencv-python scikit-learn matplotlib
python fingerprint_eval.py --data DB1_B --no-enhance
python fingerprint_eval.py --data DB1_B

## Limitations
- Small dataset (80 images), so EER estimates have low confidence
- ORB is a general-purpose detector, not minutiae-based
- No liveness / spoof detection

## Conclusion
Gabor enhancement did **not** improve ORB-based matching. The 1.5-point difference is about 4 of 280 genuine pairs, so the honest reading is "no improvement", not "clearly worse".
![Baseline](results_baseline.png)
![Enhanced](results_enhanced.png)

## Acknowledgements
Enhancement uses the open-source fingerprint_enhancer library by Utkarsh Deshmukh.
