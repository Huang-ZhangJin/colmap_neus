from typing import Tuple

import torch
import torch.nn as nn

import cneus.libcneus as _C


SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
SH_C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

class SphericalHarmonicsEncoding(nn.Module):
    """ Sphere Harmonic encoding embedding. Code was taken from https://github.com/sxyu/plenoctree. """
    def __init__(self, sh_degree: int = 4) -> None:
        super().__init__()
        assert sh_degree<=5   # not support sh_degree large than 5
        assert sh_degree>-1
        self.basis_dim = sh_degree**2
        self.sh_degree = sh_degree

    def forward(self, dirs : torch.Tensor) -> torch.Tensor:
        """
        Evaluate spherical harmonics bases at unit directions,
        without taking linear combination.
        At each point, the final result may the be
        obtained through simple multiplication.

        :param dirs: torch.Tensor (..., 3) unit directions
        :return: torch.Tensor (..., basis_dim)
        """
        result = torch.empty((*dirs.shape[:-1], self.basis_dim), dtype=dirs.dtype, device=dirs.device)
        result[..., 0] = SH_C0  # sh_degree = 0 or 1
        if self.basis_dim > 1:  # sh_degree = 2
            x, y, z = dirs.unbind(-1)
            result[..., 1] = -SH_C1 * y
            result[..., 2] = SH_C1 * z
            result[..., 3] = -SH_C1 * x
            if self.basis_dim > 4:  # sh_degree = 3
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                result[..., 4] = SH_C2[0] * xy
                result[..., 5] = SH_C2[1] * yz
                result[..., 6] = SH_C2[2] * (2.0 * zz - xx - yy)
                result[..., 7] = SH_C2[3] * xz
                result[..., 8] = SH_C2[4] * (xx - yy)

                if self.basis_dim > 9: # sh_degree = 4
                    result[..., 9] = SH_C3[0] * y * (3 * xx - yy)
                    result[..., 10] = SH_C3[1] * xy * z
                    result[..., 11] = SH_C3[2] * y * (4 * zz - xx - yy)
                    result[..., 12] = SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                    result[..., 13] = SH_C3[4] * x * (4 * zz - xx - yy)
                    result[..., 14] = SH_C3[5] * z * (xx - yy)
                    result[..., 15] = SH_C3[6] * x * (xx - 3 * yy)

                    if self.basis_dim > 16: # sh_degree = 5
                        result[..., 16] = SH_C4[0] * xy * (xx - yy)
                        result[..., 17] = SH_C4[1] * yz * (3 * xx - yy)
                        result[..., 18] = SH_C4[2] * xy * (7 * zz - 1)
                        result[..., 19] = SH_C4[3] * yz * (7 * zz - 3)
                        result[..., 20] = SH_C4[4] * (zz * (35 * zz - 30) + 3)
                        result[..., 21] = SH_C4[5] * xz * (7 * zz - 3)
                        result[..., 22] = SH_C4[6] * (xx - yy) * (7 * zz - 1)
                        result[..., 23] = SH_C4[7] * xz * (xx - 3 * yy)
                        result[..., 24] = SH_C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
        return result

    def __repr__(self) -> str:
        return f"Spherical Harmoics Encoding (sh_degree={self.sh_degree})"


class SphericalHarmonicsEncoding_NGP(nn.Module):
    def __init__(self, sh_degree: int = 4) -> None:
        super().__init__()
        assert sh_degree<=5   # not support sh_degree large than 5
        assert sh_degree>-1
        self.basis_dim = sh_degree**2
        self.sh_degree = sh_degree

    def forward(self, dirs : torch.Tensor) -> torch.Tensor:
        result = torch.empty((*dirs.shape[:-1], self.basis_dim), dtype=dirs.dtype, device=dirs.device)
        result[..., 0] = 0.28209479177387814    # sh_degree = 0 or 1
        if self.basis_dim > 1:                  # sh_degree = 2
            x, y, z = dirs.unbind(-1)
            result[..., 1] = -0.48860251190291987 * y
            result[..., 2] =  0.48860251190291987 * z
            result[..., 3] = -0.48860251190291987 * x
            if self.basis_dim > 4:              # sh_degree = 3
                x2, y2, z2 = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                result[..., 4] =  1.0925484305920792 * xy                            
                result[..., 5] = -1.0925484305920792 * yz                           
                result[..., 6] =  0.94617469575755997 * z2 - 0.31539156525251999      
                result[..., 7] = -1.0925484305920792 * xz                           
                result[..., 8] =  0.54627421529603959 * x2 - 0.54627421529603959 * y2 

                if self.basis_dim > 9:         # sh_degree = 4
                    result[..., 9]  = 0.59004358992664352 * y * (-3.0 * x2 + y2) 
                    result[..., 10] = 2.8906114426405538 * xy * z               
                    result[..., 11] = 0.45704579946446572 * y * (1.0  - 5.0 * z2)
                    result[..., 12] = 0.3731763325901154 * z * (5.0 * z2 - 3.0 ) 
                    result[..., 13] = 0.45704579946446572 * x * (1.0  - 5.0 * z2)
                    result[..., 14] = 1.4453057213202769 * z * (x2 - y2)        
                    result[..., 15] = 0.59004358992664352 * x * (-x2 + 3.0 *y2) 

                    if self.basis_dim > 16:    # sh_degree = 5
                        x4, y4, z4 = x2 * x2, y2 * y2, z2 * z2
                        result[..., 16] = 2.5033429417967046 * xy * (x2 - y2)                   
                        result[..., 17] = 1.7701307697799304 * yz * (-3.0 * x2 + y2)                      
                        result[..., 18] = 0.94617469575756008 * xy * (7.0 * z2 - 1.0)                            
                        result[..., 19] = 0.66904654355728921 * yz * (3.0  - 7.0 * z2)                  
                        result[..., 20] = -3.1735664074561294 * z2 + 3.7024941420321507 * z4 + 0.31735664074561293       
                        result[..., 21] = 0.66904654355728921 * xz * (3.0  - 7.0 * z2)
                        result[..., 22] = 0.47308734787878004 * (x2 - y2) * (7.0 * z2 - 1.0 )                              
                        result[..., 23] = 1.7701307697799304 * xz * (-x2 + 3.0 *y2)             
                        result[..., 24] = -3.7550144126950569 * x2 * y2 + 0.62583573544917614 * x4 + 0.62583573544917614 * y4
        return result

    def __repr__(self) -> str:
        return f"Spherical Harmoics Encoding NGP (sh_degree={self.sh_degree})"


class _SphericalHarmonics(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dirs: torch.Tensor, sh_degree: int) -> torch.Tensor:
        ctx.save_for_backward(dirs)
        ctx.sh_degree = sh_degree
        outputs = _C.spherical_harmonic_forward(dirs, sh_degree)

        return outputs

    @staticmethod
    def backward(ctx, outputs_grad: torch.Tensor) -> Tuple:
        (dirs, ) = ctx.saved_tensors
        sh_degree = ctx.sh_degree
        inputs_grad = _C.spherical_harmonic_backward(outputs_grad, dirs, sh_degree)

        return inputs_grad, None

sphericalharmoic = _SphericalHarmonics.apply
