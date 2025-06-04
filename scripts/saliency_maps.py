###################### TẠO SALIENCY MÁP CHO ALL DỮ LIỆU, CHỈ CẦN TRAIN CHO 1 FOLD VỚI TOÀN BỘ DỮ LIỆU LÀ ĐƯỢC
#OK
import sys
# put your path here
#sys.path.extend(['/disk/scratch2/alessandro/new_code/Dif-fuse'])
sys.path.append("..")
sys.path.append(".")
import numpy as np
from utils.arg_parsing import parse_args
from datetime import datetime
from torch.autograd import grad
import random
from autoencoder_architectures import *
import torchvision
from guided_diffusion.bratsloader import *
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_classifier,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from torch.utils.data import DataLoader
import imageio
import skimage
from utils.storage import (
    build_experiment_folder,
    restore_model,
)

################################################################################## Data


args = parse_args()

main_start = datetime.now()
dt_string = main_start.strftime("%d/%m/%Y %H:%M:%S")
print("Start main() date and time =", dt_string)

args = parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)  # set seed
random.seed(args.seed)

device = (
    torch.cuda.current_device()
    if torch.cuda.is_available()
    else "cpu"
)

args.device = device

height = 256
width = 256
channels = 4
args.num_workers = 4

val_ds = BRATSDataset(mode="test", fold=args.fold, test_flag=False)
val_loader = torch.utils.data.DataLoader(
    val_ds,
    batch_size= 1,
    shuffle=False)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

################################################################################## Model classifier Resnet50
classifier_path = f"/kaggle/input/brats20-models-fold2/modelcls020000.pt"
model = create_classifier(
    image_size=256,
    classifier_use_fp16=False,
    classifier_width=32,
    classifier_depth=4,
    classifier_attention_resolutions="32,16,8",
    classifier_use_scale_shift_norm=True,
    classifier_resblock_updown=True,
    classifier_pool="attention",
    classifier_dropout=0.0,
    dataset='brats' ## Đổi thành cái khác
)

print("loading classifier model...")
model.load_state_dict(
    dist_util.load_state_dict(classifier_path)
)

model = model.to(device)
if args.num_gpus_to_use > 1:
    model = nn.DataParallel(model)

model.eval()

#################### AE
enc_out = 512
ae = AE_no_bottleneck_6_12_16(batch_size= args.batch_size, input_height = 256, enc_type='my_resnet18', first_conv=False, maxpool1=False, enc_out_dim=enc_out, latent_dim=int(enc_out/2), lr=0.0001)

ae = ae.to(device)
if args.num_gpus_to_use > 1:
    ae = nn.DataParallel(ae)
    ae.module.encoder = nn.DataParallel(ae.module.encoder)
    ae.module.decoder = nn.DataParallel(ae.module.decoder)

autoencoder_filepath, _, _ = build_experiment_folder(
    experiment_name='autoencoder',
    log_path=args.logs_path + "/" + "epochs_" + str(args.max_epochs),

)
_ = restore_model(restore_fields={"model": ae}, path='/kaggle/input/autoencoder-acat-brats20/autoencoder', device=device, best=True)

ae.eval()

########################################################################## Optimisation


def to_tensor_grad(z, requires_grad=False):
    z = torch.Tensor(z).to(device)
    z.requires_grad=requires_grad
    return z

def to_numpy(z):
    return z.data.cpu().numpy()

alpha = 100
beta = 0.001
m = nn.Softmax(dim=-1)

################ By training 20 times, we can calculate the region of anomaly -- we train z here (ohhh) by gradient descent
def compute_counterfactual(z, z0, targets, t0, criterion_class = nn.CrossEntropyLoss(),  criterion_norm = nn.L1Loss()):
    for i in range(20):
        # print(i)
        z = to_tensor_grad(z, requires_grad=True)
        z_0 = to_tensor_grad(z0)
        logits = model(ae.decoder(z),t0)
        saliency_loss = criterion_class(input=logits, target=targets)
        distance = criterion_norm(z, z_0)
        # print('loss', saliency_loss, 'distance',distance, 'prob', m(logits)[:, 1])

        loss = saliency_loss + alpha * distance

        dl_dz = grad(loss, z)[0]
        z = z - beta * dl_dz ############ gradient descent
        z = to_numpy(z)
        # print(z)
    return to_tensor_grad(z)

def compute_saliency(z, im2):
    shifted_image_1 = ae.decoder(z)
    shifted_image_1 = shifted_image_1.detach().cpu()
    dimage = torch.abs(im2.cpu()- shifted_image_1)

    return dimage


####################### Folder lưu saliency
saliency_root = '/kaggle/working/diffusion-anomaly-3/saliency_maps'

if not os.path.exists(saliency_root):
    os.makedirs(saliency_root)

### Make saliency_maps for 4 levels one time
for loader in [val_loader]:
    for i, (inputs, _,_, _) in enumerate(loader):
            print(f"Sample {i}:")
            inputs = inputs.to(device)
            #classes = torch.randint(low=0, high=1, size=(1,), device=device)
            t0 = torch.randint(low=0, high=1, size=(1,), device=device)

            im1_enc = inputs
            im2 = ae.decoder(ae.encoder(im1_enc)).detach().to('cpu')

            z = to_numpy(ae.encoder(im1_enc))
            z0 = z

            # positive counterfactual
            targets = (torch.ones([inputs.shape[0]], dtype=torch.long)).to(device)
            z_out = compute_counterfactual(z.copy(), z0.copy(), targets, t0)
            dimage1 = compute_saliency(z_out, im2.clone())

            # negative counterfactual
            targets = (torch.zeros([inputs.shape[0]], dtype=torch.long)).to(device)
            z_out = compute_counterfactual(z.copy(), z0.copy(), targets, t0)
            dimage2 = compute_saliency(z_out, im2.clone())


            dimage1 = dimage1*(1.0 / torch.amax(dimage1, dim=(-3, -2, -1), keepdim=True))
            dimage2 = dimage2*(1.0 / torch.amax(dimage2, dim=(-3, -2, -1), keepdim=True))

            dimage = (dimage1+dimage2)/2
            dimage = dimage*(1.0 / torch.amax(dimage, dim=(-3, -2, -1), keepdim=True))

            #### Lưu một cái thôi và minmaxscaler, dùng difftot, scipy ..... rồi dùng binary của otsu
            ############## Dimage tìm ra có 4 chiều
            for j in range(inputs.shape[0]):
                for k, level in enumerate(['flair', 't1', 't2', 't1ce']):
                    path = os.path.join(saliency_root, f"sample_{i}_" + level + '.png')
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    imageio.imwrite(path, skimage.img_as_ubyte(dimage[j,k, :, :]))

########## 

