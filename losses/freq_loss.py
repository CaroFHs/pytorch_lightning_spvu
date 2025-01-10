import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Down_wt(nn.Module):
    def __init__(self):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar').to(device)

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        # x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)        

        return yL, y_HL, y_LH, y_HH
    
class hwt_loss(nn.Module):
    def __init__(self):
        super(hwt_loss, self).__init__()
        self.freq_d = Down_wt().to(device)
        self.l1 = torch.nn.L1Loss()  # L1损失
        self.mse_loss = nn.MSELoss() # 均方误差损失
    def forward(self, img_irg, img_ir):
        ## 红外真值和生成图像分别 haar wt 提取低频和高频
        irgL, irg_HL, irg_LH, irg_HH = self.freq_d(img_irg)
        irL, ir_HL, ir_LH, ir_HH = self.freq_d(img_ir)

        ## 低频做mse_loss
        yl_mseloss = self.mse_loss(irgL, irL)
        ## 高频做l1_loss
        yhl_l1loss = self.l1(irg_HL, ir_HL)
        ylh_l1loss = self.l1(irg_LH, ir_LH)
        yhh_l1loss = self.l1(irg_HH, ir_HH)

        yh_l1loss = (yhl_l1loss + ylh_l1loss + yhh_l1loss)/3 # 总高频损失

        ## 总频域损失
        total_loss = yl_mseloss + yh_l1loss

        return total_loss


    