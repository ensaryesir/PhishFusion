"""
PhishFusion Results Analysis - legitimacy_3049 Dataset
Analyzes the output from phishintention.py with TTA-enabled Swin Transformer
Focus: False Positive Rate (FPR) Analysis for Robustness
"""

import pandas as pd
import numpy as np
from collections import Counter
import json
from datetime import datetime
import os

# Dosya kontrolü
if not os.path.exists('results.txt'):
    print("❌ HATA: results.txt dosyası bulunamadı!")
    exit()

# Read results
print("📊 Loading results from results.txt...")
data = []
with open('results.txt', 'r', encoding='utf-8') as f:
    for line in f:
        try:
            parts = line.strip().split('\t')
            if len(parts) >= 7:
                folder = parts[0]
                url = parts[1]
                # PhishIntention çıktısında bazen None string olarak gelir
                phish_category = int(parts[2]) # 1: Phishing, 0: Benign
                pred_target = parts[3] if parts[3] != 'None' else None
                matched_domain = parts[4] if parts[4] != 'None' else None
                siamese_conf = float(parts[5]) if parts[5] != 'None' else 0.0
                runtime = parts[6]
                
                data.append({
                    'folder': folder,
                    'url': url,
                    'prediction': phish_category, # 1 means ALARM (False Positive for this dataset)
                    'pred_target': pred_target,
                    'matched_domain': matched_domain,
                    'confidence': siamese_conf,
                    'runtime': runtime
                })
        except Exception as e:
            print(f"⚠️ Satır okuma hatası: {e}")
            continue

df = pd.DataFrame(data)
print(f"✅ Loaded {len(df)} URLs")

# Parse runtime breakdown
def parse_runtime(runtime_str):
    try:
        parts = runtime_str.split('|')
        if len(parts) >= 5:
            return {
                'awl_time': float(parts[0]),
                'logo_time': float(parts[1]), # TTA etkisi burada
                'crp_class_time': float(parts[2]),
                'crp_locator_time': float(parts[3]),
                'url_time': float(parts[4])
            }
    except:
        pass
    return None

df['runtime_parsed'] = df['runtime'].apply(parse_runtime)
valid_runtimes = [r for r in df['runtime_parsed'] if r is not None]

# === ACADEMIC STATISTICS FOR 3049 DATASET ===
print("\n" + "="*70)
print("🧪 ROBUSTNESS ANALYSIS (3049 Misleading Legitimacy Dataset)")
print("="*70)

total_samples = len(df)
# Bu veri setinde "Phishing" (1) tahmini aslında bir hatadır (False Positive)
false_positives = df[df['prediction'] == 1]
fp_count = len(false_positives)
fp_rate = (fp_count / total_samples) * 100

print(f"\n📉 False Positive Analysis (LOWER IS BETTER):")
print(f"  Total Samples:      {total_samples}")
print(f"  False Positives:    {fp_count}")
print(f"  False Positive Rate (FPR): {fp_rate:.2f}%")
print(f"  Accuracy (Robustness):     {100 - fp_rate:.2f}%")

# Hatanın Kaynağı Analizi (Swin vs URL)
# Eğer prediction 1 ise ve pred_target (logo tahmini) varsa, hatayı Swin yapmıştır.
# Eğer prediction 1 ise ama pred_target yoksa, hatayı URL veya diğer modüller yapmıştır.
visual_fps = false_positives[false_positives['pred_target'].notna()]
visual_fp_count = len(visual_fps)
url_fps = fp_count - visual_fp_count

print(f"\n🔍 Source of False Positives:")
print(f"  Visual Model (Swin) Mistakes: {visual_fp_count} ({visual_fp_count/total_samples*100:.2f}%)")
print(f"  Other/URL Module Mistakes:    {url_fps} ({url_fps/total_samples*100:.2f}%)")

print(f"\n🎯 Wrongly Accused Brands (Top 10):")
if visual_fp_count > 0:
    wrong_brands = visual_fps['pred_target'].value_counts().head(10)
    for i, (brand, count) in enumerate(wrong_brands.items(), 1):
        print(f"  {i:2d}. {brand}: {count} times ({count/total_samples*100:.2f}%)")
else:
    print(f"  - None (Perfect Visual Robustness!)")

print(f"\n⏱️ Runtime Efficiency (Critical for TTA Evaluation):")
if valid_runtimes:
    awl_times = [r['awl_time'] for r in valid_runtimes]
    logo_times = [r['logo_time'] for r in valid_runtimes]
    crp_times = [r['crp_class_time'] for r in valid_runtimes]
    url_times = [r['url_time'] for r in valid_runtimes]
    total_avgs = sum([np.mean(awl_times), np.mean(logo_times), np.mean(crp_times), np.mean(url_times)])
    
    print(f"  Average Logo Matching Time (with TTA): {np.mean(logo_times):.4f}s")
    print(f"  Average URL Analysis Time:             {np.mean(url_times):.4f}s")
    print(f"  Average Total Latency per URL:         {total_avgs:.4f}s")
    
    print(f"\n📊 Component Breakdown:")
    print(f"  AWL Detection:       {np.mean(awl_times):.4f}s ({np.mean(awl_times)/total_avgs*100:.1f}%)")
    print(f"  Logo Matching (TTA): {np.mean(logo_times):.4f}s ({np.mean(logo_times)/total_avgs*100:.1f}%)")
    print(f"  CRP Classification:  {np.mean(crp_times):.4f}s ({np.mean(crp_times)/total_avgs*100:.1f}%)")
    print(f"  URL Analysis:        {np.mean(url_times):.4f}s ({np.mean(url_times)/total_avgs*100:.1f}%)")

# === CONFIDENCE ANALYSIS ===
print(f"\n🎯 Confidence Score Analysis (False Positives only):")
if fp_count > 0:
    fp_confidences = false_positives['confidence']
    print(f"  Mean:   {fp_confidences.mean():.4f}")
    print(f"  Median: {fp_confidences.median():.4f}")
    print(f"  Std:    {fp_confidences.std():.4f}")
    print(f"  Min:    {fp_confidences.min():.4f}")
    print(f"  Max:    {fp_confidences.max():.4f}")

# === SAVE DETAILED REPORT ===
print("\n💾 Saving detailed report...")

report = {
    "model": "Swin-Tiny + TTA + URL Hybrid",
    "dataset": "3049_misleading_legitimacy",
    "dataset_description": "All samples are LEGITIMATE websites. Any phishing detection is a FALSE POSITIVE.",
    "evaluation_date": datetime.now().isoformat(),
    "metrics": {
        "total_samples": int(total_samples),
        "false_positives": int(fp_count),
        "false_positive_rate_percent": float(fp_rate),
        "robustness_accuracy_percent": float(100 - fp_rate),
        "visual_model_fps": int(visual_fp_count),
        "visual_model_fpr_percent": float(visual_fp_count/total_samples*100),
        "url_module_fps": int(url_fps),
        "url_module_fpr_percent": float(url_fps/total_samples*100)
    },
    "wrongly_accused_brands": {k: int(v) for k, v in visual_fps['pred_target'].value_counts().head(10).items()} if visual_fp_count > 0 else {},
    "confidence_scores_fps": {
        "mean": float(fp_confidences.mean()) if fp_count > 0 else 0,
        "median": float(fp_confidences.median()) if fp_count > 0 else 0,
        "std": float(fp_confidences.std()) if fp_count > 0 else 0,
        "min": float(fp_confidences.min()) if fp_count > 0 else 0,
        "max": float(fp_confidences.max()) if fp_count > 0 else 0
    },
    "runtime_analysis": {
        "avg_logo_matching_tta_seconds": float(np.mean(logo_times)) if valid_runtimes else 0,
        "avg_url_analysis_seconds": float(np.mean(url_times)) if valid_runtimes else 0,
        "avg_total_latency_seconds": float(total_avgs) if valid_runtimes else 0
    }
}

with open('robustness_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=4, ensure_ascii=False)

print("✅ Report saved to robustness_report.json")

# === ACADEMIC PAPER SUMMARY ===
print("\n" + "="*70)
print("📝 SUMMARY FOR ACADEMIC PAPER")
print("="*70)

print(f"""
ROBUSTNESS EVALUATION RESULTS:

Dataset: legitimacy_3049 (All legitimate websites)
Objective: Measure False Positive Rate (FPR)

Key Metrics:
- Total URLs Tested: {total_samples}
- False Positive Rate: {fp_rate:.2f}%
- Robustness Accuracy: {100 - fp_rate:.2f}%

Source Analysis:
- Visual Model (Swin) FPR: {visual_fp_count/total_samples*100:.2f}%
- URL Module FPR: {url_fps/total_samples*100:.2f}%

Interpretation:
{"✅ EXCELLENT: Visual model shows high selectivity" if visual_fp_count/total_samples*100 < 5 else "⚠️ MODERATE: Visual model needs improvement"}

TTA Impact:
- Average Logo Matching Time: {np.mean(logo_times):.4f}s (with 3-view augmentation)
- Overall System Latency: {total_avgs:.4f}s per URL

For Paper:
"On the legitimacy_3049 dataset comprising 3,041 legitimate websites, 
our Swin Transformer-based model achieved a {100 - fp_rate:.2f}% robustness accuracy 
(FPR: {fp_rate:.2f}%), with visual false positives contributing only 
{visual_fp_count/total_samples*100:.2f}% of all alarms."
""")

print("="*70)
print("🎉 Analysis Complete!")
print("="*70)
