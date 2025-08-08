# mspEN

Implementing auto encoders with Matrix subspace projection for neurograms and electrodograms generated from the TIMIT corpus of speech data.
The work here corresponds to the paper: Latent Space Factorisation and Manipulation via Matrix Subspace Projection (ICML2020).
mspEN stands for `matrix subspace projection Electrodogram Neurogram`.

## Electrodogram auto-encoder

Encoder:
[Input: 2]
→ Dense(4), ReLU
→ Dense(2), ReLU
→ Dense(1) ← latent

Decoder:
→ Dense(2), ReLU
→ Dense(4), ReLU
→ Dense(2)

## Neurogram auto-encoder

Encoder:
[Input: 150]
→ Dense(64), ReLU
→ Dense(32), ReLU
→ Dense(16) ← latent

Decoder:
→ Dense(32), ReLU
→ Dense(64), ReLU
→ Dense(150), Sigmoid or Linear
