from network.AEI_Net import *
from network.MultiscaleDiscriminator import *
from utils.Dataset import FaceEmbed, With_Identity
from torch.utils.data import DataLoader
import torch.optim as optim
from face_modules.model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
import torch.nn.functional as F
import torch
import time
import torchvision
import cv2
from apex import amp
#import visdom
from torch.utils.tensorboard import SummaryWriter
from DiffAugment_pytorch import DiffAugment
import pickle

#vis = visdom.Visdom(server='127.0.0.1', env='faceshifter', port=8099)
writer = SummaryWriter('runs/FaceShifter')
batch_size = 6
lr_G = 4e-4
lr_D = 4e-4
max_epoch = 2000
show_step = 30
save_epoch = 1
model_save_path = './saved_models/'
optim_level = 'O1'
policy = 'color'

# fine_tune_with_identity = False

device = torch.device('cuda')
# torch.set_num_threads(12)

G = AEI_Net(c_id=512).to(device)
D = MultiscaleDiscriminator(input_nc=3, n_layers=6, norm_layer=torch.nn.InstanceNorm2d).to(device)
G.train()
D.train()

arcface = Backbone(50, 0.6, 'ir_se').to(device)
arcface.eval()
arcface.load_state_dict(torch.load('./face_modules/model_ir_se50.pth', map_location=device), strict=False)
arcface.requires_grad_(False)

opt_G = optim.Adam(G.parameters(), lr=lr_G, betas=(0, 0.999))
opt_D = optim.Adam(D.parameters(), lr=lr_D, betas=(0, 0.999))

G, opt_G = amp.initialize(G, opt_G, opt_level=optim_level)
D, opt_D = amp.initialize(D, opt_D, opt_level=optim_level)

try:
    G.load_state_dict(torch.load('./saved_models/G_latest.pth', map_location=torch.device('cpu')), strict=False)
    D.load_state_dict(torch.load('./saved_models/D_latest.pth', map_location=torch.device('cpu')), strict=False)
    opt_G.load_state_dict(torch.load('./saved_models/optG_latest.pth', map_location=torch.device('cpu')))
    opt_D.load_state_dict(torch.load('./saved_models/optD_latest.pth', map_location=torch.device('cpu')))
    amp.load_state_dict(torch.load('./saved_models/amp_latest.pth', map_location=torch.device('cpu')))
except Exception as e:
    print(e)

# if not fine_tune_with_identity:
    # dataset = FaceEmbed(['../celeb-aligned-256_0.85/', '../ffhq_256_0.85/', '../vgg_256_0.85/', '../stars_256_0.85/'], same_prob=0.5)
# else:
    # dataset = With_Identity('../washed_img/', 0.8)
FaceSources = ['/home/olivier/Images/FaceShifter/celeba-256/', '/home/olivier/Images/FaceShifter/Perso/', '/home/olivier/Images/FaceShifter/VGGFaceTrain/', '/home/olivier/Images/FaceShifter/FFHQ/']
dataset = FaceEmbed(FaceSources, same_prob=0.2)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)


MSE = torch.nn.MSELoss()
L1 = torch.nn.L1Loss()


def hinge_loss(X, positive=True):
    if positive:
        return torch.relu(1-X).mean()
    else:
        return torch.relu(X+1).mean()


def get_grid_image(X):
    X = X[:8]
    X = torchvision.utils.make_grid(X.detach().cpu(), nrow=X.shape[0]) * 0.5 + 0.5
    return X


def make_image(Xs, Xt, Y):
    Xs = get_grid_image(Xs)
    Xt = get_grid_image(Xt)
    Y = get_grid_image(Y)
    return torch.cat((Xs, Xt, Y), dim=1).numpy()


# prior = torch.FloatTensor(cv2.imread('./prior.png', 0).astype(np.float)/255).to(device)

print(torch.backends.cudnn.benchmark)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
for epoch in range(0, max_epoch):
    # torch.cuda.empty_cache()
    for iteration, data in enumerate(dataloader):
        niter = epoch * len(dataloader) + iteration
        start_time = time.time()
        Xs, Xt, same_person = data
        Xs = Xs.to(device)
        Xt = Xt.to(device)
        # embed = embed.to(device)
        with torch.no_grad():
            #embed, Xs_feats = arcface(F.interpolate(Xs[:, :, 19:237, 19:237], [112, 112], mode='bilinear', align_corners=True))
            embed, _ = arcface(F.interpolate(Xs[:, :, 19:237, 19:237], [112, 112], mode='bilinear', align_corners=True))
        same_person = same_person.to(device)
        #diff_person = (1 - same_person)

        # train G
        D.requires_grad_(False)
        opt_G.zero_grad()
        Y, Xt_attr = G(Xt, embed)

        Di = D(DiffAugment(Y, policy=policy))
        L_adv = 0

        for di in Di:
            #L_adv += hinge_loss(di[0], True)
            L_adv -= di[0].mean()
        

        Y_aligned = Y[:, :, 19:237, 19:237]
        #ZY, Y_feats = arcface(F.interpolate(Y_aligned, [112, 112], mode='bilinear', align_corners=True))
        ZY, _ = arcface(F.interpolate(Y_aligned, [112, 112], mode='bilinear', align_corners=True))
        L_id =(1 - torch.cosine_similarity(embed, ZY, dim=1)).mean()

        Y_attr = G.get_attr(Y)
        L_attr = 0
        for i in range(len(Xt_attr)):
            #L_attr += torch.mean(torch.pow(Xt_attr[i] - Y_attr[i], 2).reshape(batch_size, -1), dim=1).mean()
            L_attr += torch.mean(torch.pow(Xt_attr[i] - Y_attr[i], 2))
        L_attr /= 2.0

        #L_rec = torch.sum(0.5 * torch.mean(torch.pow(Y - Xt, 2).reshape(batch_size, -1), dim=1) * same_person) / (same_person.sum() + 1e-6)
        L_rec = MSE(Y[same_person], Xt[same_person]) * same_person.sum() /(2.0 * batch_size)

        lossG = 1*L_adv + 10*L_attr + 5*L_id + 10*L_rec
        # lossG = 1*L_adv + 10*L_attr + 5*L_id + 10*L_rec
        with amp.scale_loss(lossG, opt_G) as scaled_loss:
            scaled_loss.backward()

        # lossG.backward()
        opt_G.step()

        # train D
        D.requires_grad_(True)
        opt_D.zero_grad()
        # with torch.no_grad():
        #     Y, _ = G(Xt, embed)
        fake_D = D(DiffAugment(Y.detach(), policy=policy))
        loss_fake = 0
        for di in fake_D:
            loss_fake += hinge_loss(di[0], False)

        true_D = D(DiffAugment(Xs, policy=policy))
        loss_true = 0
        for di in true_D:
            loss_true += hinge_loss(di[0], True)
        # true_score2 = D(Xt)[-1][0]

        lossD = 0.5*(loss_true.mean() + loss_fake.mean())

        with amp.scale_loss(lossD, opt_D) as scaled_loss:
            scaled_loss.backward()
        # lossD.backward()
        opt_D.step()
        batch_time = time.time() - start_time
        if iteration % show_step == 0:
            image = make_image(Xs, Xt, Y)
            #vis.image(image[::-1, :, :], opts={'title': 'result'}, win='result')
            #cv2.imwrite('./gen_images/latest.jpg', image.transpose([1,2,0]))
            writer.add_image('Train/Xs Xt Y', image[::-1, :, :], niter)
            writer.add_scalars('Train/Generator losses',
                    {'L_adv': L_adv.item(), 'L_id': L_id.item(),
                        'L_attr': L_attr.item(), 'L_rec': L_rec.item()},
                    niter)
            writer.add_scalars('Train/Adversarial losses',
                    {'Generator': lossG.item(), 'Discriminator': lossD.item()},
                    niter)
        print(f'epoch: {epoch}    {iteration} / {len(dataloader)}')
        print(f'lossD: {lossD.item()}    lossG: {lossG.item()} batch_time: {batch_time}s')
        print(f'L_adv: {L_adv.item()} L_id: {L_id.item()} L_attr: {L_attr.item()} L_rec: {L_rec.item()}')
        if iteration % 1000 == 0:
            torch.save(G.state_dict(), './saved_models/G_latest.pth')
            torch.save(D.state_dict(), './saved_models/D_latest.pth')
            torch.save(opt_D.state_dict(), './saved_models/optG_latest.pth')
            torch.save(opt_D.state_dict(), './saved_models/optD_latest.pth')
            torch.save(amp.state_dict(), './saved_models/amp_latest.pth')
            with open('./saved_models/niter.pkl', 'wb') as f:
                pickle.dump(niter, f)
        if (niter + 1) % 10000 == 0:
            torch.save(G.state_dict(), f'./saved_models/G_iteration_{niter + 1}.pth')
            torch.save(D.state_dict(), f'./saved_models/D_iteration_{niter + 1}.pth')
            with open(f'./saved_models/niter_{niter + 1}.pkl', 'wb') as f:
                pickle.dump(niter, f)



