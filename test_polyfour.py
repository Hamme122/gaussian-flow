import torch
import taichi as ti
from polyfourier import DDDMModel 

ti.init(arch=ti.cuda)

class Args:
    def __init__(self, no_dx=False, no_ds=False, no_dr=False, no_do=False, no_dshs=False):
        self.no_dx = no_dx
        self.no_ds = no_ds
        self.no_dr = no_dr
        self.no_do = no_do
        self.no_dshs = no_dshs

def test_dddm_model():
    
    args = Args(no_dx=False, no_ds=False, no_dr=False, no_do=False, no_dshs=True)
    model = DDDMModel(type_name="poly_fourier", feat_dim=32, poly_factor=1.0, Hz_factor=1.0, args=args)

    batch_size = 3200
    feat_dim = 16
    output_dim = 14

    means3D = torch.randn(batch_size, 3).cuda()
    scales = torch.randn(batch_size, 3).cuda()
    rotations = torch.randn(batch_size, 4).cuda()
    opacity = torch.randn(batch_size, 1).cuda()
    shs = torch.randn(batch_size, 1, 3).cuda()
    
    time = 0.33
    time = torch.tensor(time).to(means3D.device).repeat(means3D.shape[0],1)

    dddmpara = torch.nn.Parameter(torch.randn((batch_size, feat_dim, output_dim, 3), device='cuda', requires_grad=True))

    # Define loss function and optimizer
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam([dddmpara], lr=0.001)
    target = torch.zeros(batch_size, output_dim-3).cuda()

    num_epochs = 4000
    for epoch in range(num_epochs):

        output = model.forward(means3D, scales, rotations, opacity, shs, time, dddmpara, feat_dim)
        means3D_out, scales_out, rotations_out, opacity_out, shs_out = output

        # Combine outputs if necessary or use one for simplicity
        combined_output = torch.cat((means3D_out, scales_out, rotations_out, opacity_out), dim=1)
        
        loss = loss_fn(combined_output, target)

        optimizer.zero_grad()  
        loss.backward(retain_graph=True)
        optimizer.step()  
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    print("means3D_out:", means3D_out)
    print("scales_out:", scales_out)
    print("rotations_out:", rotations_out)
    print("opacity_out:", opacity_out)
    print("shs_out:", shs_out)

if __name__ == "__main__":
    test_dddm_model()
