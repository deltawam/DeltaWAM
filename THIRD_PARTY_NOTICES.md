# Third-party notices

DeltaWAM contains a minimal vendored subset of StarWAM under `starwam/`.
The subset provides ActionDiT, flow-matching utilities, the scheduler,
Wan-style transformer blocks, checkpoint loading, and Wan2.2 initialization.
It is distributed under the Apache License 2.0; the corresponding license is
preserved at `third_party/starwam/LICENSE`.

The DeltaTok and DeltaWorld implementation derives from the DeltaTok project.
Its original attribution is preserved in `NOTICE`.

External model and benchmark assets are not redistributed:

- DINOv3 weights are obtained from the official gated model distribution.
- T5-Base is obtained from Hugging Face.
- Wan2.2 weights are obtained from their official distribution.
- LIBERO datasets and simulator assets must be obtained from LIBERO.
- DeltaWAM checkpoints are distributed separately and are not part of Git.
