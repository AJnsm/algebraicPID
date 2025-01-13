# The Fast Möbius transform: An algebraic approach to decompose information
This repository contains the code associated to the paper [The Fast Möbius transform: An algebraic approach to decompose information](https://arxiv.org/abs/2410.06224) by Abel Jansma, Pedro Mediano, and Fernando Rosas. The notebook [algebraic_infoDecomp.ipynb](algebraic_infoDecomp.ipynb) contains the code to calculate the Möbius function for up to 5 variables, as well as the associated PID calculations on example distributions. It also shows how to use the Möbius function to calculate the 5-variable PID. The precomputed Möbius functions are stored as sparse scipy arrays (and `csv`s for $n\leq 4$) in the `FMT_outputs/` directory, together with `csv`s that contain the list of antichains, in the same ordering as the Möbius function entries. 

The brain data analysis can be reproduced with the [brain_analysis.ipynb](brain_analysis.ipynb) notebook, and the Baroque music analysis with [music_analysis.ipynb](music_analysis.ipynb).

To cite this work, please use

```
@article{jansma2024fast,
  title={The Fast M{\"o}bius Transform: An algebraic approach to information decomposition},
  author={Jansma, Abel and Mediano, Pedro AM and Rosas, Fernando E},
  journal={arXiv preprint arXiv:2410.06224},
  year={2024}
}
```
