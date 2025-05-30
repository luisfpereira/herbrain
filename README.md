# 🧠 HerBrain
HerBrain uses AI for quantifying changes in the female brain during menstruation, pregnancy, and menopause.

[![DOI](https://zenodo.org/badge/827974965.svg)](https://zenodo.org/doi/10.5281/zenodo.13356094)


## 🎬 HerBrain App Demo ##

[![Demo](/images/HerBrainDemo_thumbnail.png)](https://youtu.be/GW7zc48lrTs)

## 🎤 Our Public Talk on Womens' Brain Health and AI ##

[![BBB Talk](/images/bbb_thumbnail.png)](https://youtu.be/BsdNQUcwb1M)
Visual on thumbnail slide taken from: Caitlin M Taylor, Laura Pritschet, and Emily G Jacobs. The scientific body of knowledge–whose body does it serve? A spotlight on oral contraceptives and women’s health factors in neuroimaging. Frontiers in neuroendocrinology, 60:100874, 2021.

## 🤖 Installing HerBrain

```bash
pip install git+https://github.com/geometric-intelligence/herbrain.git@main#egg=herbrain[app]
```

Or the classic pipeline: `git clone` and:
```bash
conda create -n herbrain python=3.11
conda activate herbrain
pip install .[app]
```


## 🏃‍♀️ How to Run the Code ##

Once you install the package and download the data, run

```bash
python -m herbrain <app-name>
```

You can always get some help by doing

```bash
python -m herbrain --help
```


## 🌎 Bibtex ##
If this code is useful to your research, please cite:

```
@misc{myers2023geodesic,
      title={Geodesic Regression Characterizes 3D Shape Changes in the Female Brain During Menstruation},
      author={Adele Myers and Caitlin Taylor and Emily Jacobs and Nina Miolane},
      year={2023},
      eprint={2309.16662},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## 👩‍🔧 Authors ##
[Adele Myers](https://ahma2017.wixsite.com/adelemyers)

[Nina Miolane](https://www.ninamiolane.com/)
