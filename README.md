# IOPNet: Profiling Individual Opinions in Image Quality Assessment

Code repository for the IOPNet individual opinion profiling model.

---

## 📄 Paper

Yueran Ma, Huasheng Wang, Jiang Liu, Padraig Corcoran, Roger M. Whitaker, Stuart M. Allen, Gualtiero B. Colombo, Wei Zhou, Guangtao Zhai, Ying Chen, Hantao Liu  
**IOPNet: Profiling Individual Opinions in Image Quality Assessment**  
IEEE Transactions on Multimedia (TMM), 2026

IEEE Xplore page:  
[https://ieeexplore.ieee.org/abstract/document/11718033](https://ieeexplore.ieee.org/abstract/document/11718033)

DOI:  
[https://doi.org/10.1109/TMM.2026.3739488](https://doi.org/10.1109/TMM.2026.3739488)

---

## Network Architecture

The model combines a Vision Transformer backbone, Transposed Attention Blocks and Swin Transformer blocks to predict individual quality scores. The training implementation uses fully connected score and weight branches and produces 19 observer scores for CUID.

## Dataset

The experiments use the **Cardiff University Image quality Database (CUID)** introduced by **Lucie Lévêque et al.**, with individual ratings from 19 observers. CUID is a third-party dataset. This repository contains training and data-loading code only; CUID images and individual opinion scores are not distributed here. Running the training commands requires access to both the images and the individual-score annotations. Mean opinion scores alone are insufficient for training the individual opinion prediction model.

For dataset availability and access, please refer to the original authors' official announcements and distribution channels.

If you use CUID, please also cite the dataset paper:

L. Lévêque, J. Yang, X. Yang, P. Guo, K. Dasalla, L. Li, Y. Wu, and H. Liu, ["CUID: A New Study of Perceived Image Quality and its Subjective Assessment,"](https://doi.org/10.1109/ICIP40778.2020.9190737) *2020 IEEE International Conference on Image Processing (ICIP)*, pp. 116-120, 2020. DOI: 10.1109/ICIP40778.2020.9190737.

For users with access to the data, place images in `<data-root>/stimuli/` and specify the observer-score workbook with `--annotations`. The expected workbook format contains filenames in row 1, observer ratings in rows 2-20, a separator in row 21, and MOS in row 22. The first column contains row labels. Scores are read without rescaling.

The default image extension is `.bmp`. Use `--image-suffix ""` to keep the extensions in the workbook.

---

## USAGE

Train:

```bash
python main.py --data-root /path/to/cuid/all --annotations /path/to/CUID_Ind_score.xlsx --output-dir results/run1
```

Resume training:

```bash
python main.py --data-root /path/to/cuid/all --annotations /path/to/CUID_Ind_score.xlsx --resume results/run1/checkpoint.pth.tar --output-dir results/run1 --epochs 100
```

Evaluate:

```bash
python main.py --data-root /path/to/cuid/all --annotations /path/to/CUID_Ind_score.xlsx --evaluate results/run1/model_best.pth.tar
```

Use `--no-pretrain` to disable ImageNet weight downloads and `--device cpu` or `--device cuda` to select the device. Run `python main.py --help` for all options.

Training uses a seeded 9:1 image-level split. Checkpoints and training logs are saved to `--output-dir`; the best checkpoint is selected by validation RMSE.

## Requirements

Python 3.10 or later.

```bash
pip install -r requirements.txt
```

## Citation

```bibtex
@article{ma2026iopnet,
  title   = {{IOPNet}: Profiling Individual Opinions in Image Quality Assessment},
  author  = {Ma, Yueran and Wang, Huasheng and Liu, Jiang and Corcoran, Padraig and
             Whitaker, Roger M. and Allen, Stuart M. and Colombo, Gualtiero B. and
             Zhou, Wei and Zhai, Guangtao and Chen, Ying and Liu, Hantao},
  journal = {IEEE Transactions on Multimedia},
  year    = {2026},
  doi     = {10.1109/TMM.2026.3739488},
  url     = {https://doi.org/10.1109/TMM.2026.3739488}
}
```

## Acknowledgment

Our code incorporates components and design ideas inspired by [MANIQA](https://github.com/IIGROUP/MANIQA), proposed by Sidi Yang et al. We thank the authors for making their code publicly available.

MANIQA paper: [MANIQA: Multi-Dimension Attention Network for No-Reference Image Quality Assessment](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/html/Yang_MANIQA_Multi-Dimension_Attention_Network_for_No-Reference_Image_Quality_Assessment_CVPRW_2022_paper.html), CVPR Workshops, 2022.

We also acknowledge Swin Transformer, timm, PyTorch and torchvision for the components and libraries used in this work.

## License

This project is licensed under the [MIT License](LICENSE). Third-party dependencies are covered by their respective licenses.
