Official PyTorch implementation of "MiT Loss: Medical Image-aware Transfer-calibrated Loss for Enhanced Classification".

Temperature-based entropy regularization methods have evolved from fixed-temperature entropy maximization ([TET Loss](https://github.com/JEFfersusu/TET_loss)), to learnable global temperature with entropy constraints ([MiT Loss](https://github.com/JEFfersusu/MiT_loss)).
However, existing approaches rely on a single global temperature and unstable entropy targets, limiting their effectiveness under class imbalance.

This CCT-EAL extends this line of work by introducing class-conditional temperatures and a stable entropy alignment objective, where the predictive entropy is softly aligned with an EMA-estimated label entropy.
This design enables end-to-end calibration during training, avoids dual-variable instability, and better accommodates class-dependent uncertainty.

### Comparison of Temperature–Entropy Based Training Losses

| Method   | Temperature Modeling | Entropy Target        | Weighting Strategy | Stability |
|----------|----------------------|-----------------------|--------------------|-----------|
| TET Loss     | Fixed                | Higher is better      | Fixed λ            | ❌        |
| MiT Loss      | Globally learnable    | Empirical label entropy | Dual update        | ⚠️        |
| **CCT-EAL** | **Class-conditional learnable** | **EMA label entropy** | **Warm-up α**      | ✅        |
