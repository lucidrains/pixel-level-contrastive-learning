<img src="./propagate.png"></img>

## Pixel-level Contrastive Learning (wip)

Implementation of Pixel-level Contrastive Learning, proposed in the paper <a href="https://arxiv.org/abs/2011.10043">"Propagate Yourself"</a>, in Pytorch. In addition to doing contrastive learning on the pixel level, the online network further passes the pixel level representations to a Pixel Propagation Module and enforces a similarity loss to the target network. They beat all previous unsupervised and supervised methods in segmentation tasks.

## Citations

```bibtex
@misc{xie2020propagate,
    title={Propagate Yourself: Exploring Pixel-Level Consistency for Unsupervised Visual Representation Learning}, 
    author={Zhenda Xie and Yutong Lin and Zheng Zhang and Yue Cao and Stephen Lin and Han Hu},
    year={2020},
    eprint={2011.10043},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
