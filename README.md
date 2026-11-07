# ACDC myocardial radiomics

This code reproduces the analysis for the AIDD8122 report on the incremental value of short-axis cine CMR myocardial radiomics for ACDC disease classification.

## Reproduce the analysis

Use Python 3.14 and obtain ACDC through the [official data page](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html). The database directory must contain `training/patient*` and `testing/patient*` with `Info.cfg`, phase NIfTI images, and the matching reference masks. The published split contains 100 development and 50 test cases.

```text
python -m pip install -r requirements.txt
python study.py --database /path/to/database --out results/summary.json
```

The JSON contains aggregate cohort summaries, model metrics, class-level confusion matrices, a paired bootstrap interval for the primary contrast, and mask perturbation results. The script uses reference masks to measure radiomics and conventional features, and fixes the seeds for fold construction and bootstrap sampling. No dataset files are distributed with this repository.

## Citation

Bernard O, Lalande A, Zotti C, et al. Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation and Diagnosis: Is the Problem Solved? IEEE Transactions on Medical Imaging. 2018;37(11):2514–2525. [doi:10.1109/TMI.2018.2837502](https://doi.org/10.1109/TMI.2018.2837502).
