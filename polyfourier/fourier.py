import torch
import taichi as ti
from taichi.math import vec3, pi, vec4, vec2
        
class Fourier(torch.nn.Module):
    def __init__(self, max_degree, Hz_base_factor=1):
        super(Fourier,self).__init__()
        self.max_degree = max_degree
        Hz_base = Hz_base_factor * pi
        @ti.kernel
        def fft_kernel_fwd(
            factors: ti.types.ndarray(dtype=vec2, ndim=3), 
            # t: ti.types.ndarray(), 
            t: float,
            out: ti.types.ndarray(), 
            degree: int
        ):
            for pid, dim_id, f_id in ti.ndrange(
                factors.shape[0], 
                factors.shape[2],
                # max_degree
            ):
                out_sum = 0.
                for f_id in ti.static(range(max_degree)):
                    if f_id < degree:
                        fac_vec = factors[pid, f_id, dim_id]
                        # noise_vec = noise[pid, dim_id, f_id]
                        current_w = (f_id) * 2 * Hz_base * t
                        x = current_w #* fac_vec[2] + fac_vec[3]
                        sin = fac_vec[0] * ti.sin(x)
                        cos = fac_vec[1] * ti.cos(x)
                        out_sum += sin + cos
                out[pid, dim_id] = out_sum
        
        self.fft_kernel_fwd = fft_kernel_fwd
                
        @ti.kernel
        def fft_kernel_bwd(
            d_factors: ti.types.ndarray(), 
            factors: ti.types.ndarray(dtype=vec2, ndim=3),
            # t: ti.types.ndarray(), 
            t: float,
            d_out: ti.types.ndarray(), 
            degree: int
        ):
            for pid, dim_id, f_id in ti.ndrange(
                d_factors.shape[0], 
                d_factors.shape[2],
                max_degree
            ):
                # if f_id < degree:
                # for f_id in ti.static(range(max_degree)):
                if f_id < degree:
                    # fac_vec = factors[pid, f_id, dim_id]
                    current_w = (f_id) * 2 * Hz_base * t
                    d_o = d_out[pid, dim_id]
                    # noise_vec = noise[pid, dim_id, f_id]
                    x = current_w #* fac_vec[2] + fac_vec[3]
                    d_factors[pid, f_id, dim_id, 0] = d_o*ti.sin(x)
                    d_factors[pid, f_id, dim_id, 1] = d_o*ti.cos(x)
                    # sin_df = fac_vec[0] * ti.cos(x)
                    # cos_df = -fac_vec[1] * ti.sin(x)
                    # d_term = d_o * (sin_df + cos_df)
                    # d_factors[pid, f_id, dim_id, 2] = d_term * current_w
                    # d_factors[pid, f_id, dim_id, 3] = d_term
                else:
                    d_factors[pid, f_id, dim_id, 0] = 0.0
                    d_factors[pid, f_id, dim_id, 1] = 0.0
                    # d_factors[pid, f_id, dim_id, 2] = 0.0
                    # d_factors[pid, f_id, dim_id, 3] = 0.0
                        
                        
        self.fft_kernel_bwd = fft_kernel_bwd   
        class _fft_taichi(torch.autograd.Function):
            @staticmethod
            def forward(ctx, factors, t, degree=1):
                ctx.save_for_backward(factors)
                ctx.t = t
                ctx.degree = degree
                out = torch.zeros(
                    (factors.shape[0], factors.shape[2]), 
                    dtype=torch.float32, 
                    device=factors.device
                )
                self.fft_kernel_fwd(
                    factors, 
                    t, 
                    out, 
                    # noise.contiguous(), 
                    degree
                )
                return out
            
            @staticmethod
            def backward(ctx, d_out):
                factors, = ctx.saved_tensors
                t = ctx.t
                degree = ctx.degree
                d_factors = torch.empty_like(factors)
                # make sure contiguous 
                d_out = d_out.contiguous()
                self.fft_kernel_bwd(
                    d_factors, 
                    factors,
                    t, 
                    d_out,
                    # noise, 
                    degree
                )
                return d_factors, None, None
                
        self._module_function = _fft_taichi.apply
        
    def forward(self, factors, timestamp, degree):
        return self._module_function(
            factors.contiguous(), 
            timestamp, 
            # noise,
            degree,
        ) / self.max_degree