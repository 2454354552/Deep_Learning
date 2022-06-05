import torch
import torchvision
from torch import nn
import time
import torch
import torch.nn.functional as F
from torch import optim
from torch import nn
from einops import rearrange


#残差模块，放在每个前馈网络和注意力之后
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        y = self.fn(x, **kwargs)
        out = y + x
        return out

#layernorm归一化,放在多头注意力层和激活函数层。用绝对位置编码的BERT，layernorm用来自身通道归一化
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        out = self.fn(x)
        return out



#放置多头注意力后，因为在于多头注意力使用的矩阵乘法为线性变换，后面跟上由全连接网络构成的FeedForward增加非线性结构
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)
#多头注意力层，多个自注意力连起来。使用qkv计算
class Attention(nn.Module):
    def __init__(self, dim, heads=8):    #dim=64, heads=8
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads   #  *x.shape(100,17,64)

        qkv = self.to_qkv(x)   #  qkv.shape(100,17,192)

        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)  # q.shape(100,8,17,8) k.shape(100,8,17,8) v.shape(100,8,17,8)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale  #(100,8,17,17)

        # if mask is not None:
        #     mask = F.pad(mask.flatten(1), (1, 0), value = True)
        #     assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
        #     mask = mask[:, None, :] * mask[:, :, None]
        #     dots.masked_fill_(~mask, float('-inf'))
        #     del mask

        attn = dots.softmax(dim=-1)  #attn.shape:(100,8,17,17)


        out = torch.einsum('bhij,bhjd->bhid', attn, v)  #out.shape:(100,8,17,8)

        out = rearrange(out, 'b h n d -> b n (h d)')  # out.shape(100,17,64)

        out =  self.to_out(out)  #用一个全连接层做分类，唯独不变，输出(100,17,64)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):  #dim=64, depth=6, heads=8, mlp_dim=128
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x
#将图像切割成一个个图像块,组成序列化的数据输入Transformer执行图像分类任务。
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))   #(1,17,64)
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim)   #dim=64, depth=6, heads=8, mlp_dim=128

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, img, mask=None):
        p = self.patch_size   # p=7  将图像分成4x4,每个patch 7x7 共4x4=16个patch

        #输入的img大小(100,1,28,28)
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)  # x.shape(100,16,49)
        x = self.patch_to_embedding(x)  #x.shape(100,16,64)



        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        #cls_tokens.shape:(100,1,64)


        x = torch.cat((cls_tokens, x), dim=1)  #x.shape:(100,17,64)
        x += self.pos_embedding  #x.shape:(100,17,64)

        x = self.transformer(x, mask)  #x.shape(100,17,64)

        #x[:, 0].shape(100,64)
        x = self.to_cls_token(x[:, 0])  #x.shape(100,64)  相当于取了所有patch的第0行所有数

        return self.mlp_head(x)  #返回维度:(100,10)



torch.manual_seed(42)

DOWNLOAD_PATH = './data'
BATCH_SIZE_TRAIN = 100
BATCH_SIZE_TEST = 1000

transform_mnist = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.1307,), (0.3081,))])

train_set = torchvision.datasets.MNIST(DOWNLOAD_PATH, train=True, download=True,
                                       transform=transform_mnist)




train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

test_set = torchvision.datasets.MNIST(DOWNLOAD_PATH, train=False, download=True,
                                      transform=transform_mnist)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE_TEST, shuffle=True)

def train_epoch(model, optimizer, data_loader, loss_history):
    total_samples = len(data_loader.dataset)
    model.train()

    for i, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=1)
        # print('output shape:{}, target.shape:{}'.format(output.shape, target.shape))  [100,10]  [100]
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            loss_history.append(loss.item())


def evaluate(model, data_loader, loss_history):
    model.eval()

    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for data, target in data_loader:
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)

            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)
    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')

N_EPOCHS = 1

start_time = time.time()
'''
patch大小为 7x7（对于 28x28 图像，这意味着每个图像 4 x 4 = 16 个patch）、10 个可能的目标类别（0 到 9）和 1 个颜色通道（因为图像是灰度）。
在网络参数方面，使用了 64 个单元的维度，6 个 Transformer 块的深度，8 个 Transformer 头，MLP 使用 128 维度。'''
model = ViT(image_size=28,
            patch_size=7,
            num_classes=10,
            channels=1,
            dim=64,
            depth=6,
            heads=8,
            mlp_dim=128)

optimizer = optim.Adam(model.parameters(), lr=0.003)

train_loss_history, test_loss_history = [], []
for epoch in range(1, N_EPOCHS + 1):
    print('Epoch:', epoch)
    train_epoch(model, optimizer, train_loader, train_loss_history)
    evaluate(model, test_loader, test_loss_history)

print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')