# EP55 — Neural Architecture Search & AutoDL Explained

**Deep Learning Series · Episode 55 · Module 11**

Can an algorithm design a better neural network than a human?  
We run a real AutoKeras search on the digits dataset and publish the **honest** results.

## Key Results (5 seeds)

| Model              | Mean Test Acc | Range          | Parameters          |
|--------------------|---------------|----------------|---------------------|
| AutoKeras best     | 98.06%        | 97.22 – 99.17  | 42,926 – 1,187,853  |
| Hand-made CNN      | 98.22%        | 98.06 – 98.33  | 19,466              |

→ **A tie** (gap < 1 test image).

Of 60 architectures searched: **34 never learned**, 15 intermediate, **11 reached ≥95%**.

## Quick Start

```bash
pip install tensorflow-cpu==2.20.0 keras==3.10.0 autokeras==3.0.0 scikit-learn numpy
python ep55_autokeras.py
