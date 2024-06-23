import torch
from typing import Callable
from .poly import Polynomial
from .fourier import Fourier
from .poly_fourier import PolyFourier


class DDDMModel(torch.nn.Module):
    def __init__(self, type_name: str = "poly", feat_dim: int = 32, poly_factor: float = 1.0, Hz_factor: float = 1.0, args = None):
        super(DDDMModel, self).__init__()
        self.type_name = type_name
        self.poly_factor = poly_factor
        self.Hz_factor = Hz_factor
        self.feat_dim = feat_dim
        self.create_model()
        self.args = args
        
    def create_model(self):
        if self.type_name == "fourier":
            self.trajectory_func = Fourier(
                self.feat_dim,
                Hz_base_factor=self.Hz_factor
            )
        elif self.type_name == "poly_fourier":
            self.trajectory_func = PolyFourier(
                self.feat_dim,
                poly_base_factor=self.poly_factor,
                Hz_base_factor=self.Hz_factor
            )
        elif self.type_name == "poly":
            self.trajectory_func = Polynomial(
                self.feat_dim,
                poly_base_factor=self.poly_factor,
            )
        else:
            self.trajectory_func = None
            print("Trajectory type not found")
    
    def forward(self, means3D, scales, rotations, opacity, shs, time, dddmpara = None, feat_degree = 16):
        
        if self.trajectory_func:
            delta = self.trajectory_func(dddmpara, time, feat_degree)    
        else:
            raise ValueError("Trajectory function not properly initialized")
        
        return self.deformation_dynamic(means3D, scales, rotations, opacity, shs, delta)

    def deformation_dynamic(self, means3D, scales, rotations, opacity, shs, delta):
        # Clone the inputs to avoid in-place modifications
        new_means3D = means3D.clone()
        new_scales = scales.clone()
        new_rotations = rotations.clone()
        new_opacity = opacity.clone()
        new_shs = shs.clone()

        if self.args.no_dx == False:
            dx = delta[:, :3]
            new_means3D += dx

        if self.args.no_ds == False:            
            ds = delta[:, 3:6]
            new_scales += ds

        if self.args.no_dr == False:
            dr = delta[:, 6:10]
            new_rotations += dr

        if self.args.no_do == False:
            do = delta[:, 10:11]
            new_opacity += do

        if self.args.no_dshs == False:
            dshs = delta[:, 11:14]
            new_shs[:, 0, :3] += dshs

        return new_means3D, new_scales, new_rotations, new_opacity, new_shs

