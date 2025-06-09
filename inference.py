import torch
import numpy as np
import os
import open3d as o3d
import argparse
import pdb
import yaml
from tools.models.model_LEMON_d import LEMON as LEMON_d
from tools.models.model_LEMON_p import LEMON as LEMON_p
from tools.models.model_LEMON_d_laso import LEMON as LEMON_laso
from tools.models.LEMON_noCur import LEMON_wocur
from PIL import Image
from dataset_utils.dataset_3DIR import _3DIR
from tools.utils.build_layer import create_mesh, build_smplh_mesh, Pelvis_norm
from tools.utils.evaluation import visual_pred, generate_proxy_sphere
from dataset_utils.dataset_3DIR import img_normalize, pc_normalize
from tools.utils.mesh_sampler import get_sample
from torch.utils.data import DataLoader

def inference_batch(opt, dict, val_loader, model, device, model_type='d'):

    checkpoint = torch.load(dict['best_checkpoint'], map_location=device)

    model.load_state_dict(checkpoint)
    model = model.to(device)
    model = model.eval()
    contact_color = np.array([255.0, 191.0, 0.])

    def save_path(path):
        file_name = path.split('/')[-1]
        obj, aff = file_name.split('_')[0], file_name.split('_')[1]
        hm_save_folder = dict['contact_result_folder'] + obj + '/' + aff
        spatial_folder = dict['spatial_result_folder'] + obj
        if not os.path.exists(hm_save_folder):
            os.makedirs(hm_save_folder)
        if not os.path.exists(spatial_folder):
            os.makedirs(spatial_folder)
        file_name = file_name.split('.')[0] + '.ply'
        hm_save_file = os.path.join(hm_save_folder, file_name)
        spatial_save_file = os.path.join(spatial_folder, file_name)
        return hm_save_file, spatial_save_file
    
    with torch.no_grad():
        for i, data_info in enumerate(val_loader):
            B = data_info['hm_curvature'].size(0)
            img_paths = data_info['img_path']
            pts_paths = data_info['Pts_path']
            H, face = build_smplh_mesh(data_info['human'])
            H = H.to(device)
            H, pelvis = Pelvis_norm(H, device)
            O = data_info['Pts'].float().to(device)
            text_desc = data_info.get('text_desc', None)
            C_h = data_info['hm_curvature'].to(device)
            C_o = data_info['obj_curvature'].to(device)
            if model_type == 'no_cur':
                pre_contact, pre_affordance, pre_spatial, _ = model(O, H, text_desc=text_desc)
            elif model_type == 'laso':
                dummy_img = torch.zeros((O.size(0), 3, 224, 224), device=device)
                outputs = model(dummy_img, O, H, C_h, C_o, text_desc=text_desc)
                pre_contact, pre_affordance, pre_spatial = outputs[0], outputs[1], outputs[2]
            elif model_type == 'p':
                dummy_img = torch.zeros((O.size(0), 3, 224, 224), device=device)
                outputs = model(dummy_img, O, H, C_h, C_o, text_desc=text_desc)
                pre_contact, pre_affordance, pre_spatial = outputs[0], outputs[1], outputs[2]
            else:
                pre_contact, pre_affordance, pre_spatial, _ = model(O, H, C_h, C_o, text_desc=text_desc)
            pre_affordance = pre_affordance.cpu().detach().numpy()
            contact_fine = pre_contact[-1]

            for j in range(B):
                vertices = H[j].detach().cpu().numpy()
                spatial_center = pre_spatial[j].detach().cpu().numpy()
                spatial_sphere = generate_proxy_sphere(spatial_center, pts_paths[j])
                colors = np.array([255.0, 255.0, 255.0])[None, :].repeat(6890, axis=0)
                contact_id = torch.where(contact_fine[j] > 0.5)[0].cpu()
                contact_id = np.asarray(contact_id)
                colors[contact_id] = contact_color
                colors = colors / 255.0

                contact_mesh = create_mesh(vertices=vertices, faces=face, colors=colors)
                mesh_save_path, spatial_save_path = save_path(data_info['img_path'][j])
                o3d.io.write_triangle_mesh(mesh_save_path, contact_mesh)
                o3d.io.write_triangle_mesh(spatial_save_path, spatial_sphere)                
                visual_pred(img_paths[j], pre_affordance[j], pts_paths[j], dict['affordance_result_folder'])

def mask_img(img_path, mask_path):
    img = Image.open(img_path).convert('RGB')
    mask = Image.open(mask_path).convert('RGB')
    img, mask = np.asarray(img), np.asarray(mask)
    back_ground = np.array([0, 0, 0])
    mask_bi = np.all(mask == back_ground, axis=2)
    mask_img = np.ones_like(mask)
    mask_img[mask_bi] = back_ground
    masked_img = img * mask_img
    masked_img = Image.fromarray(masked_img)
    return masked_img

def extract_point_file(path):
    with open(path,'r') as f:
        coordinates = []
        lines = f.readlines()
    for line in lines:
        line = line.strip('\n')
        line = line.strip(' ')
        data = line.split(' ')
        coordinate = [float(x) for x in data]
        coordinates.append(coordinate)
    data_array = np.array(coordinates)
    points_coordinates = data_array[:, 0:3]
    affordance_label = data_array[: , 3:]

    return points_coordinates, affordance_label

def read_yaml(path):
    file = open(path, 'r', encoding='utf-8')
    string = file.read()
    dict = yaml.safe_load(string)
    return dict

def get_human_param(path):
    smplh_param = {}
    param_data = np.load(path, allow_pickle=True)
    smplh_param['shape'] = torch.tensor(param_data['shape']).unsqueeze(0)
    smplh_param['transl'] = torch.tensor(param_data['transl']).unsqueeze(0)
    smplh_param['body_pose'] = torch.tensor(param_data['body_pose']).reshape(1, 21, 3, 3)
    smplh_param['left_hand_pose'] = torch.tensor(param_data['left_hand_pose']).reshape(1, 15, 3, 3)
    smplh_param['right_hand_pose'] = torch.tensor(param_data['right_hand_pose']).reshape(1, 15, 3, 3)
    smplh_param['global_orient'] = torch.tensor(param_data['global_orient']).reshape(1, 3, 3)

    return smplh_param

def inference_single(model, opt, dict, device, outdir):

    checkpoint = torch.load(dict['best_checkpoint'], map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model = model.eval()



    #load human
    human_param = get_human_param(opt.human_param_path)
    vertices, face = build_smplh_mesh(human_param)
    mesh_sampler = get_sample(device=None)
    hm_curvature = np.load(opt.C_h, allow_pickle=True)
    hm_curvature = torch.from_numpy(hm_curvature).to(torch.float32)
    C_h = mesh_sampler.downsample(hm_curvature).unsqueeze(0).to(device)
    vertices = vertices.to(device)
    H, pelvis = Pelvis_norm(vertices, device)
    H = H.to(device)

    #load object
    Pts, affordance_label = extract_point_file(opt.object)
    Pts = pc_normalize(Pts)
    Pts = Pts.transpose()
    O = torch.from_numpy(Pts).float().unsqueeze(0).to(device)
    C_o = np.load(opt.C_o, allow_pickle=True)
    C_o = torch.from_numpy(C_o).to(torch.float32).unsqueeze(dim=-1).unsqueeze(dim=0).to(device)
    if isinstance(model, LEMON_wocur):
        pre_contact, pre_affordance, pre_spatial, _ = model(O, H, text_desc=None)
    elif isinstance(model, LEMON_laso):
        dummy_img = torch.zeros((1, 3, 224, 224), device=device)
        outputs = model(dummy_img, O, H, C_h, C_o, text_desc=None)
        pre_contact, pre_affordance, pre_spatial = outputs[0], outputs[1], outputs[2]
    elif isinstance(model, LEMON_p):
        dummy_img = torch.zeros((1, 3, 224, 224), device=device)
        outputs = model(dummy_img, O, H, C_h, C_o, text_desc=None)
        pre_contact, pre_affordance, pre_spatial = outputs[0], outputs[1], outputs[2]
    else:
        pre_contact, pre_affordance, pre_spatial, _ = model(O, H, C_h, C_o, text_desc=None)
    contact_fine = pre_contact[-1]
    pre_affordance = pre_affordance[0].cpu().detach().numpy()

    #save
    contact_color = np.array([255.0, 191.0, 0.])
    vert = H.detach().cpu().numpy()
    spatial_center = pre_spatial[0].detach().cpu().numpy()
    spatial_sphere = generate_proxy_sphere(spatial_center, opt.object)
    colors = np.array([255.0, 255.0, 255.0])[None, :].repeat(6890, axis=0)
    contact_id = torch.where(contact_fine[0] > 0.5)[0].cpu()
    contact_id = np.asarray(contact_id)
    colors[contact_id] = contact_color
    colors = colors / 255.0

    contact_mesh = create_mesh(vertices=vert[0], faces=face, colors=colors)
    mesh_save_path = os.path.join(outdir, opt.img_path.split('/')[-1].split('.')[0]+'_contact.ply')
    spatial_save_path = os.path.join(outdir, opt.img_path.split('/')[-1].split('.')[0]+'_spatial.ply')
    o3d.io.write_triangle_mesh(mesh_save_path, contact_mesh)
    o3d.io.write_triangle_mesh(spatial_save_path, spatial_sphere)

    reference_color = np.array([255, 0, 0])
    back_color = np.array([190, 190, 190])
    pred_point = o3d.geometry.PointCloud()
    pred_point.points = o3d.utility.Vector3dVector(O[0].detach().cpu().numpy().transpose())
    pred_color = np.zeros((O.shape[2],3))
    for i, pred in enumerate(pre_affordance):
        scale_i = pred
        pred_color[i] = (reference_color-back_color) * scale_i + back_color
    pred_point.colors = o3d.utility.Vector3dVector(pred_color.astype(np.float64) / 255.0)
    object_save_path = os.path.join(outdir, opt.img_path.split('/')[-1].split('.')[0]+'_object.ply')
    o3d.io.write_point_cloud(object_save_path, pred_point)

def inference_single_wo_curvature(model, opt, dict, device, outdir):

    checkpoint = torch.load(dict['best_checkpoint'], map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model = model.eval()



    #load human
    human_param = get_human_param(opt.human_param_path)
    vertices, face = build_smplh_mesh(human_param)
    vertices = vertices.to(device)
    H, pelvis = Pelvis_norm(vertices, device)
    H = H.to(device)
    #load object
    Pts, affordance_label = extract_point_file(opt.object)
    Pts = pc_normalize(Pts)
    Pts = Pts.transpose()
    O = torch.from_numpy(Pts).float().unsqueeze(0).to(device)
    pre_contact, pre_affordance, pre_spatial, _ = model(O, H, text_desc=None)
    contact_fine = pre_contact[-1]
    pre_affordance = pre_affordance[0].cpu().detach().numpy()

    #save
    contact_color = np.array([255.0, 191.0, 0.])
    vert = H.detach().cpu().numpy()
    spatial_center = pre_spatial[0].detach().cpu().numpy()
    spatial_sphere = generate_proxy_sphere(spatial_center, opt.object)
    colors = np.array([255.0, 255.0, 255.0])[None, :].repeat(6890, axis=0)
    contact_id = torch.where(contact_fine[0] > 0.5)[0].cpu()
    contact_id = np.asarray(contact_id)
    colors[contact_id] = contact_color
    colors = colors / 255.0

    contact_mesh = create_mesh(vertices=vert[0], faces=face, colors=colors)
    mesh_save_path = os.path.join(outdir, opt.img_path.split('/')[-1].split('.')[0]+'_contact.ply')
    spatial_save_path = os.path.join(outdir, opt.img_path.split('/')[-1].split('.')[0]+'_spatial.ply')
    o3d.io.write_triangle_mesh(mesh_save_path, contact_mesh)
    o3d.io.write_triangle_mesh(spatial_save_path, spatial_sphere)

    reference_color = np.array([255, 0, 0])
    back_color = np.array([190, 190, 190])
    pred_point = o3d.geometry.PointCloud()
    pred_point.points = o3d.utility.Vector3dVector(O[0].detach().cpu().numpy().transpose())
    pred_color = np.zeros((O.shape[2],3))
    for i, pred in enumerate(pre_affordance):
        scale_i = pred
        pred_color[i] = (reference_color-back_color) * scale_i + back_color
    pred_point.colors = o3d.utility.Vector3dVector(pred_color.astype(np.float64) / 255.0)
    object_save_path = os.path.join(outdir, opt.img_path.split('/')[-1].split('.')[0]+'_object.ply')
    o3d.io.write_point_cloud(object_save_path, pred_point)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=12, help='batch_size')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu to run')
    parser.add_argument('--yaml', type=str, default='config/infer.yaml', help='infer setting')
    #single
    parser.add_argument('--img_path', type=str, default='Demo/Backpack_carry_demo.jpg', help='single test image')
    parser.add_argument('--mask', type=str, default='Demo/Backpack_carry_mask.png', help='single test mask')
    parser.add_argument('--human_param_path', type=str, default='Demo/Backpack_human_demo.npz', help='single test human')
    parser.add_argument('--object', type=str, default='Demo/Backpack_object_demo.txt', help='single test object')
    parser.add_argument('--C_o', type=str, default='Demo/Backpack_curvature.pkl', help='single test object curvature')
    parser.add_argument('--C_h', type=str, default='Demo/Human_curvature.pkl', help='single test object curvature')
    parser.add_argument('--outdir', type=str, default='Demo/output1', help='single test ouput dir')

    opt = parser.parse_args()
    dict = read_yaml(opt.yaml)
    if opt.use_gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model_type = dict['model']
    if model_type == 'laso':
        model = LEMON_laso(dict['emb_dim'], run_type='infer', device=device)
    elif model_type == 'no_cur':
        model = LEMON_wocur(dict['emb_dim'], run_type='infer', device=device)
    elif model_type == 'p':
        model = LEMON_p(dict['emb_dim'], run_type='infer', device=device)
    else:
        model = LEMON_d(dict['emb_dim'], run_type='infer', device=device)
    #batch
    infer_type = dict['infer_type']
    if infer_type == 'batch':
        val_dataset = _3DIR(dict['val_image'], dict['val_pts'], dict['human_3DIR'],
                            dict['behave'], mode='val',
                            desc_file=dict.get('desc_file'))
        val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8)
        inference_batch(opt, dict, val_loader, model, device, model_type)
    elif infer_type == 'single':
        if model_type == 'no_cur':
            inference_single_wo_curvature(model, opt, dict, device, opt.outdir)
        else:
            inference_single(model, opt, dict, device, opt.outdir)
