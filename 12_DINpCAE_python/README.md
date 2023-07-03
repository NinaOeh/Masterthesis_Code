# Extension of DINCAE v2 model, with partial convolutions in first U-NET


DINCAE (Data-Interpolating Convolutional Auto-Encoder, Alexander Bath et al) is a neural network to
reconstruct missing data in satellite observations. Two versions of DINCAE have been released of which the first (DINCAEv1) is described in the following open access paper: https://doi.org/10.5194/gmd-13-1609-2020 and has been released using tensorflow python on this gihub page: https://github.com/gher-uliege/DINCAE and the second version (DINCAEv2) is described in this paper: https://doi.org/10.5194/gmd-15-2183-2022 and has been released using julia on this gthub page: https://github.com/gher-ulg/DINCAE.jl*.

In this reporsitory the existing DINCAEv2 model has been extended and partial convolutions have been applied in the first U-NET of the model. 