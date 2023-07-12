from Models.ModelUtils import *


class Additional(nn.Module):
    def __init__(self):
        pass

    def forward(self, x1, x2, x3):
        prediction = torch.cat((x1, x2, x3), dim=1)
        return prediction


class Resnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = ConvNormAct(3, 32, k=3, s=2, p=1)

        self.conv2 = ConvNormAct(32, 64, k=3, s=2, p=1)

        self.conv3 = ConvNormAct(64, 128, k=3, s=2, p=1)

        self.conv4 = ConvNormAct(128, 256, k=3, s=2, p=1)

        self.conv5 = ConvNormAct(256, 512, k=3, s=2, p=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        prediction = self.conv5(x)
        return prediction   # 1 3 5 7 9



class ResnetChanged(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = ConvNormAct(3, 32, k=3, s=2, p=1)

        self.conv2 = ConvNormAct(32, 64, k=3, s=2, p=1)

        self.insert_1 = ConvNormAct(64, 64, k=3, s=1, p=1)

        self.conv3 = ConvNormAct(64, 128, k=3, s=2, p=1)

        self.insert_2 = ConvNormAct(128, 128, k=3, s=1, p=1)

        self.conv4 = ConvNormAct(128, 256, k=3, s=2, p=1)

        self.conv5 = ConvNormAct(256, 512, k=3, s=2, p=1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.insert_1(x)
        x = self.conv3(x)
        x = self.insert_2(x)
        x = self.conv4(x)
        prediction = self.conv5(x)
        return prediction



if __name__ == '__main__':
    from torch.optim import Adam

    student_model = ResnetChanged()
    state_dict = torch.load('Models/Resnet.pth') # {"conv1.weight": ..., "conv1.bias": ...}
    student_model.load_state_dict(state_dict, strict=False)

    teacher_model = Resnet()
    state_dict = torch.load('Models/Resnet.pth') # {"conv1.weight": ..., "conv1.bias": ...}
    teacher_model.load_state_dict(state_dict, strict=False)
    teacher_model.eval()

    # training phase 1: 让学生模型fit教师模型

    # List[Dict["params", 想要更新的模块]]
    optimizer = Adam([{"params": student_model.insert_1.parameters()}, {"params": student_model.insert_2.parameters()}], lr=1e-3)
    loss_fn = nn.MSELoss()

    data = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        target = teacher_model(data)

    optimizer.zero_grad()
    output = student_model(data)
    loss = loss_fn(output, target)

    optimizer.step()
    torch.save(student_model.state_dict(), 'Models/Best_student.pth')


    # train phase 2: 让学生模型学习到更多的知识

    student_model = ResnetChanged()
    state_dict = torch.load('Models/Best_student.pth')  # {"conv1.weight": ..., "conv1.bias": ...}
    student_model.load_state_dict(state_dict, strict=False)

    optimizer = Adam(student_model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    data = torch.randn(1, 3, 224, 224)
    target = torch.zeros(1, 512, 7, 7)

    optimizer.zero_grad()
    output = student_model(data)
    loss = loss_fn(output, target)

    optimizer.step()


