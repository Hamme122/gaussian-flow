import torch
import taichi as ti
from taichi.math import vec2, pi, vec3
               
class Polynomial(torch.nn.Module):
    def __init__(self, max_degree, poly_base_factor=1.):
        super(Polynomial,self).__init__()
        self.max_degree = max_degree
        @ti.kernel
        def poly_kernel_fwd(
        factors: ti.types.ndarray(),
        t_array: ti.types.ndarray(), 
        out: ti.types.ndarray(), 
        degree: int
        ):
            for pid, dim_id, f_id in ti.ndrange(
                factors.shape[0], 
                factors.shape[2],
                max_degree
            ):
                    t = t_array[pid, 0]
                    if f_id < degree:
                        f = factors[pid, f_id, dim_id]
                        x = poly_base_factor * t 
                        out[pid, dim_id] += f * ti.pow(x, f_id)
                        
        self.poly_kernel_fwd = poly_kernel_fwd
                
        @ti.kernel
        def poly_kernel_bwd(
            d_factors: ti.types.ndarray(),
            factors: ti.types.ndarray(), 
            t_array: ti.types.ndarray(), 
            d_time: ti.types.ndarray(),
            d_out: ti.types.ndarray(), 
            degree: int
        ):
            for pid, dim_id, f_id in ti.ndrange(
                d_factors.shape[0], 
                d_factors.shape[2],
                max_degree
            ):
                t = t_array[pid, 0]
                if f_id < degree:
                    f = factors[pid, f_id, dim_id]
                    x = poly_base_factor * t #* f[1] + f[2]
                    d_o = d_out[pid, dim_id]
                    d_factors[pid, f_id, dim_id] = d_o * ti.pow(x, f_id)
                    d_poly_dt = f * ti.pow(x, f_id - 1) * poly_base_factor
                    d_time[pid, 0] += d_o * f_id * d_poly_dt
                else:
                    d_factors[pid, f_id, dim_id] = 0.0

        self.poly_kernel_bwd = poly_kernel_bwd
        class _polynomial_taichi(torch.autograd.Function):
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
                self.poly_kernel_fwd(factors, t, out, degree)
                return out
            
            @staticmethod
            def backward(ctx, d_out):
                factors, = ctx.saved_tensors
                t = ctx.t
                degree = ctx.degree
                d_factors = torch.empty_like(factors)
                d_time = torch.zeros_like(t)
                d_out = d_out.contiguous()
                self.poly_kernel_bwd(d_factors, factors, t, d_time, d_out, degree)
                return d_factors, d_time, None
                    
        self._module_function = _polynomial_taichi.apply
        
    def forward(self, factors, timestamp, degree):
        return self._module_function(
            factors.contiguous(), 
            timestamp.contiguous(),
            degree,
        )