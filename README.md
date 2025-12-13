# [NeurIPS 2025] Domain-RAG: Retrieval-Guided Compositional Image Generation for Cross-Domain Few-Shot Object Detection

[📚 Paper (NeurIPS 2025)](https://arxiv.org/abs/2506.05872) | [📂 Project Page](https://yuli-cs.net/papers/domain-rag) | [📦 Dataset Scripts](#dataset-preparation) | [🚀 Quick Start](#quick-start) | [🎥 Video](#video) | [✉️ Contact](#contact)

---

**Domain-RAG** is a novel retrieval-augmented generative framework designed for **Cross-Domain Few-Shot Object Detection (CD-FSOD)**. We leverage large-scale vision-language models (GroundingDINO), a curated COCO-style retrieval corpus, and Flux-based background generation to synthesize diverse, domain-aware training data that enhances FSOD generalization under domain shift.

<p align="center">
  <img src="assets/framework.svg" alt="DomainRAG Pipeline" width="700"/>
</p>

---

## ✨ Highlights

- 🔍 **Retrieval-Augmented Generation**: retrieve semantically similar source images for novel-class prompts.
- 🎨 **Flux-Redux Integration**: compose diverse backgrounds with target foregrounds for domain-aligned generation.
- 📦 **Support for Multiple Target Domains**: ArTAXOr, Clipart1k, DIOR, DeepFish, UODD, NEU-DET, and more.
- 🧪 **Strong Benchmarks**: surpasses GroundingDINO baseline in few-shot setting across CD-FSOD & RS-FSOD & CAMO-FS.

---

## 🔧 Installation

```bash
git clone https://github.com/LiYu0524/Domain-RAG.git
cd Domain-RAG
conda create -n domainrag python=3.10
conda activate domainrag
pip install -r requirements.txt
```


## Dataset Preparation

You can prepare CD-FSOD with [CDVITO](https://github.com/lovelyqian/CDFSOD-benchmark?tab=readme-ov-file)

For NWPU VHR-10(RS-FSOD) dataset, you can download it from [NWPU VHR-10](https://gcheng-nwpu.github.io/)

For CAMO-FS dataset, you can download it from [CAMO-FS](https://www.kaggle.com/datasets/danhnt/camo-fs-dataset)

## Quick start 

You can refer to `./domainrag.sh`



## Video

Walkthrough video(Chinese version): [Watch here](https://www.bilibili.com/video/BV1YznKzkEEK/?spm_id_from=333.337.search-card.all.click&vd_source=23bede4ceb3dc1ea2ffc645933850555)

## Contact

For questions and collaboration, please contact:

- Yu Li : `<liyu24@m.fudan.edu.cn>`
- Xingyu Qiu :  `<24210980129@m.fudan.edu.cn>`
- Yuqian Fu :  `<yuqian.fu@insait.ai>`


## Citation

If you find **Domain-RAG** useful in your research, please cite:

```bibtex
@article{li2025domain,
  title={Domain-RAG: Retrieval-Guided Compositional Image Generation for Cross-Domain Few-Shot Object Detection},
  author={Li, Yu and Qiu, Xingyu and Fu, Yuqian and Chen, Jie and Qian, Tianwen and Zheng, Xu and Paudel, Danda Pani and Fu, Yanwei and Huang, Xuanjing and Van Gool, Luc and others},
  journal={arXiv preprint arXiv:2506.05872},
  year={2025}
}
```

If you find **CD-Vito** useful in your research, please cite:
```bibtex
@inproceedings{fu2024cross,
  title={Cross-domain few-shot object detection via enhanced open-set object detector},
  author={Fu, Yuqian and Wang, Yu and Pan, Yixuan and Huai, Lian and Qiu, Xingyu and Shangguan, Zeyu and Liu, Tong and Fu, Yanwei and Van Gool, Luc and Jiang, Xingqun},
  booktitle={European Conference on Computer Vision},
  pages={247--264},
  year={2024},
  organization={Springer}
}
```
