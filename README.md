# Attention-enhanced DBNet for traffic scene text detection

This PyTorch reference implementation was reconstructed from Section 2 and Figures 1–4 of the manuscript after the reported experiments. It implements SCAF, adaptive pyramid fusion, and differentiable binarization for new training and evaluation runs. Software validation covers synthetic fixtures. Benchmark verification of Tables 1–4 remains pending.

## Installation

Use Python 3.12. Run commands from the repository root in an activated virtual environment.

~~~sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
~~~

## Data preparation

| Dataset | Source | Inputs |
|---|---|---|
| ICDAR 2015 | [RRC Challenge 4](https://rrc.cvc.uab.es/?ch=4) | Task 4.1 images and quadrilateral labels |
| ICDAR 2013 | [RRC Challenge 2](https://rrc.cvc.uab.es/?ch=2) | ICDAR 2013 localization images and rectangular labels |
| CTST-1600 | [He et al., IEEE Access, 2019](https://doi.org/10.1109/ACCESS.2019.2936540) | Images and polygon labels from the authorized dataset copy |

Use separate train and test splits. Place ICDAR images in the directories below and label each image with its corresponding ground-truth file, for example `img_1.jpg` and `gt_img_1.txt`. The ICDAR 2015 test image manifest preserves the official image IDs for export.

~~~sh
python -m attndb prepare-icdar icdar2015 --images data/icdar2015/train/images --labels data/icdar2015/train/gt --out data/icdar2015/train.json
python -m attndb prepare-icdar icdar2015 --images data/icdar2015/test/images --out data/icdar2015/test.json
python -m attndb prepare-icdar icdar2013 --images data/icdar2013/train/images --labels data/icdar2013/train/gt --out data/icdar2013/train.json
python -m attndb prepare-icdar icdar2013 --images data/icdar2013/test/images --labels data/icdar2013/test/gt --out data/icdar2013/test.json
~~~

For CTST-1600, store the training and test image lists with their polygon annotations in `data/ctst1600/train.json` and `data/ctst1600/test.json`. Use the JSON structure below. Image paths are relative to the manifest. Vertices follow the polygon boundary in the original image coordinates. Set `ignore` to true for regions excluded from supervision and scoring. Use an empty `instances` list for annotated images without text.

~~~json
[
  {
    "id": "sign_001",
    "image": "images/sign_001.jpg",
    "instances": [
      {"points": [[12, 15], [95, 15], [95, 40], [12, 40]], "ignore": false}
    ]
  }
]
~~~

## Training

| Variant | SCAF | Adaptive pyramid fusion |
|---|---|---|
| baseline | Off | Off |
| feature | Off | On |
| scaf | On | Off |
| full | On | On |

~~~sh
python -m attndb train --config configs/icdar2015.json --manifest data/icdar2015/train.json --variant baseline --out runs/icdar2015/baseline
python -m attndb train --config configs/icdar2015.json --manifest data/icdar2015/train.json --variant feature --out runs/icdar2015/feature
python -m attndb train --config configs/icdar2015.json --manifest data/icdar2015/train.json --variant scaf --out runs/icdar2015/scaf
python -m attndb train --config configs/icdar2015.json --manifest data/icdar2015/train.json --variant full --out runs/icdar2015/full
python -m attndb train --config configs/icdar2013.json --manifest data/icdar2013/train.json --variant full --out runs/icdar2013/full
python -m attndb train --config configs/ctst1600.json --manifest data/ctst1600/train.json --variant full --out runs/ctst1600/full
~~~

The configuration files specify the reference training recipe, including the 640 × 640 input, AdamW optimizer and fixed seed. ICDAR 2013 uses the 500 epochs stated in the manuscript. ICDAR 2015 and CTST-1600 use 100 epochs as a reference default. The model uses a two-convolution stem, a four-level pyramid, channel-axis dilation on global descriptors, and adjacent-level fusion with max and average pooling in the finer branch. These details specify this implementation where the manuscript leaves the settings open.

Each run writes the effective configuration and environment to `run.json`, training loss to `loss.jsonl`, and the model checkpoint to `last.pt`.

## Inference and evaluation

~~~sh
python -m attndb predict --checkpoint runs/icdar2015/full/last.pt --manifest data/icdar2015/test.json --out runs/icdar2015/full/predictions.json
python -m attndb export-icdar2015 --manifest data/icdar2015/test.json --predictions runs/icdar2015/full/predictions.json --out runs/icdar2015/full/rrc_task4_1.zip
python -m attndb predict --checkpoint runs/icdar2013/full/last.pt --manifest data/icdar2013/test.json --out runs/icdar2013/full/predictions.json
python -m attndb evaluate --manifest data/icdar2013/test.json --predictions runs/icdar2013/full/predictions.json --out runs/icdar2013/full/local_metrics.json
python -m attndb predict --checkpoint runs/ctst1600/full/last.pt --manifest data/ctst1600/test.json --out runs/ctst1600/full/predictions.json
python -m attndb evaluate --manifest data/ctst1600/test.json --predictions runs/ctst1600/full/predictions.json --out runs/ctst1600/full/local_metrics.json
~~~

Use the official RRC Task 4.1 evaluator for the ICDAR 2015 ZIP. The local evaluator uses score-ordered one-to-one polygon IoU matching and reports precision, recall and F-measure. Its default IoU threshold is 0.6 for CTST-1600 and 0.5 for ICDAR diagnostics. Use the ICDAR 2013 edition of the RRC evaluator for protocol-specific comparisons.

## References

- M. Liao et al., “Real-Time Scene Text Detection with Differentiable Binarization,” AAAI, 2020. [DOI](https://doi.org/10.1609/aaai.v34i07.6812).
- D. Karatzas et al., “ICDAR 2015 Competition on Robust Reading,” ICDAR, 2015. [DOI](https://doi.org/10.1109/ICDAR.2015.7333942).
- D. Karatzas et al., “ICDAR 2013 Robust Reading Competition,” ICDAR, 2013. [DOI](https://doi.org/10.1109/ICDAR.2013.221).
- X. He et al., “HTSTL: Head-and-Tail Search Network With Scale-Transfer Layer for Traffic Sign Text Detection,” IEEE Access, 2019. [DOI](https://doi.org/10.1109/ACCESS.2019.2936540).
