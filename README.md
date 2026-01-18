**Official PyTorch implementation of "Learning Class-Conditional Temperature with Entropy Alignment for Medical Image Classification".**

Temperature-based entropy regularization methods have evolved from fixed-temperature entropy maximization ([TET Loss](https://github.com/JEFfersusu/TET_loss)), to learnable global temperature with entropy constraints ([MiT Loss](https://github.com/JEFfersusu/MiT_loss)).
However, existing approaches rely on a single global temperature and unstable entropy targets, limiting their effectiveness under class imbalance.

CCT-EAL extends this line of work by introducing class-conditional temperatures and a stable entropy alignment objective, where the predictive entropy is softly aligned with an EMA-estimated label entropy.
This design enables end-to-end calibration during training, avoids dual-variable instability, and better accommodates class-dependent uncertainty.

### Comparison of Temperature–Entropy Based Training Losses

| Method   | Temperature Modeling | Entropy Target        | Weighting Strategy | Stability |
|----------|----------------------|-----------------------|--------------------|-----------|
| TET Loss     | Fixed                | Higher is better      | Fixed λ            | ❌        |
| MiT Loss      | Globally learnable    | Empirical label entropy | Dual update        | ⚠️        |
| **CCT-EAL** | **Class-conditional learnable** | **EMA label entropy** | **Warm-up α**      | ✅        |

**Note: TET Loss, MiT Loss, and CCT-EAL were introduced across different publications due to practical constraints and evolving research focus. Although they were not evaluated in a single unified study, all three methods have been independently published. We plan to further consolidate and refine this line of work in future research.**

## Citation
```bibtex
If you think that our work is useful to your research, please cite using this BibTeX😊:
@article{CCTEAL_2026,
  title={Learning Class-Conditional Temperature with Entropy Alignment for Medical Image Classification},
  author={Weichao Pan and Xu Wang and Ruida Liu and Chengze Lv and Puyu Han and Luqi Gong},
  journal={ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026},
  organization={IEEE}
}
```

If you have any questions, please contact: panweichao01@outlook.com.

