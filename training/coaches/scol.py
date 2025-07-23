import os
import torch
from tqdm import tqdm
from configs import paths_config, hyperparameters, global_config
from training.coaches.base_coach import BaseCoach
from utils.log_utils import log_images_from_w
from criteria import identity_loss
import h5py
import numpy as np
import wandb
import torch.nn.functional as F
from PIL import Image

class SCOL(BaseCoach):
    def __init__(self, data_loader, use_wandb):
        super().__init__(data_loader, use_wandb)
    
    
    def sort_img(self, qf, gf):
        query = qf.view(-1,1)
        score = torch.mm(gf,query)
        score = score.squeeze(1).cpu()
        score = score.detach().numpy()
        # predict index
        index = np.argsort(score)  #from small to large
        index = index[::-1]
        return index, score[index]

    def train(self):
        w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
        os.makedirs(w_path_dir, exist_ok=True)
        os.makedirs(f'{w_path_dir}/{paths_config.results_keyword}', exist_ok=True)

        use_ball_holder = True

        model_structure = identity_loss.Backbone(50, 0.6, 'ir_se').to(global_config.device)
        FRmodel = identity_loss.load_network(model_structure)

        # Change to test mode
        FRmodel = FRmodel.eval()
        FRmodel = FRmodel.to(global_config.device)

        for p in FRmodel.parameters():
            p.requires_grad = False

        file_path = './VGGface2_hq_Gallery_e4e.h5'
        with h5py.File(file_path, 'r') as f:
            gallery_feature = torch.FloatTensor(f['gallery_e4e'][:])
            gallery_name = [name.decode('utf-8') for name in f['gallery_name'][:]]
            gallery_feature = gallery_feature.cuda()
            lcnorm = torch.norm(gallery_feature, p=2, dim=2, keepdim=True)
            gallery_feature_norm = gallery_feature.div(lcnorm.expand_as(gallery_feature))
        
        for fname, image,img_path in tqdm(self.data_loader):
            n, c, h, w = image.size()
            lc = torch.FloatTensor(n,18,512).zero_().cuda()
            fused_latentcode = torch.FloatTensor(n,18,512).zero_().cuda()
            fused_latentcode_bottom = torch.FloatTensor(n,18,512).zero_().cuda()
            image_name = fname[0]
            id_number = img_path[0].replace("\\", "/").split("/")[-2]
            log_images_counter = 0
            real_images_batch = image.to(global_config.device)

            self.restart_training()

            if self.image_counter >= hyperparameters.max_images_to_invert:
                break

            embedding_dir = f'{w_path_dir}/{paths_config.results_keyword}/{image_name}'
            os.makedirs(embedding_dir, exist_ok=True)

            w_plus = self.get_e4e_inversion(image)
            w_plus = w_plus.to(global_config.device)
            torch.save(w_plus, f'{embedding_dir}/0.pt')

            lc += w_plus
            lcnorm = torch.norm(lc, p=2, dim=2, keepdim=True)
            lc = lc.div(lcnorm.expand_as(lc))
            for j in range(18):
                current_lc = lc[:, j, :]
                current_gallary = gallery_feature_norm[:, j, :]

                index, score = self.sort_img(current_lc, current_gallary)
                top_indices = index[:10]
                bottom_index = index[-1]
                mask_with_indices = [i for i in top_indices if gallery_name[i].split('\\')[-2] != id_number]
                valid_index = mask_with_indices[0]
                selected_style_code = gallery_feature[:, j, :][valid_index]
                selected_style_code_bottom = gallery_feature[:, j, :][bottom_index]
                fused_latentcode[:,j,:] = selected_style_code*score[0] + w_plus[0][j]*(1-score[0])
                fused_latentcode_bottom[:,j,:] = selected_style_code_bottom
            
            if self.use_wandb:
                log_images_from_w([fused_latentcode], self.G, [image_name+'_fused_G'])
            
            loss_lpips = 10
            obf_img = None
            for i in tqdm(range(hyperparameters.max_optimization_steps)):

                generated_images = self.forward(fused_latentcode)
                if i==0:
                    obf_img = generated_images.clone().detach()
                    if self.use_wandb:
                        log_images_from_w([fused_latentcode_bottom], self.G, [image_name+'_obf_bottom'])

                if loss_lpips <= hyperparameters.LPIPS_value_threshold:
                    break

                else: 
                    loss, l2_loss_val, loss_lpips, id_loss_val = self.calc_loss_obf(FRmodel, generated_images, real_images_batch, image_name, self.G, use_ball_holder, fused_latentcode, i,obf_img)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0

                if self.use_wandb and log_images_counter % global_config.image_rec_result_log_snapshot == 0:
                    log_images_from_w([fused_latentcode], self.G, [image_name+'_intermediate'])
                global_config.training_step += 1
                log_images_counter += 1

            if self.use_wandb:
                log_images_from_w([fused_latentcode_bottom], self.G, [image_name+'_obf_bottom_tunedG'])
                log_images_from_w([fused_latentcode], self.G, [image_name+'_fused_tunedG'])

            generated_images = self.G.synthesis(fused_latentcode, noise_mode='const', force_fp32=True).to(global_config.device)
            generated_images.clone().detach().to(device=global_config.device, dtype=torch.float32).requires_grad_(True)
            w_refined = self.get_e4e_inversion(generated_images)
            w_refined = w_refined.to(global_config.device)
    
            w_refined_flat = w_refined.view(1, -1)
            fused_latentcode_flat = fused_latentcode.view(1, -1)
            w_plus_flat = w_plus.view(1, -1)

            AC = torch.abs(w_refined_flat - fused_latentcode_flat)  
            IC = torch.abs(w_refined_flat - w_plus_flat) 
            id_mask = ((AC/IC)<=hyperparameters.tau).float()
            id_mask = id_mask.view(1, 18, 512)

            ap_preserved_lc = id_mask * fused_latentcode + (1 - id_mask) * w_plus
            ap_aligned_images = self.G.synthesis(ap_preserved_lc, noise_mode='const', force_fp32=True).to(global_config.device)
            w_opp = fused_latentcode_bottom.to(global_config.device)

            for param in self.G.parameters():
                param.requires_grad = False
            self.G.eval()

            real_images_batch = image.to(global_config.device)
            x_adv = ap_aligned_images.clone().detach().to(device=global_config.device, dtype=torch.float32).requires_grad_(True)

            lower_bound = torch.clamp(ap_aligned_images - hyperparameters.epsilon/255.0, min=-1.0)
            upper_bound = torch.clamp(ap_aligned_images + hyperparameters.epsilon/255.0, max=1.0)
            momentum = 1.0
            alpha = 1.0            
            for _ in range(hyperparameters.epsilon):
                attack_num += 1
                x_adv = x_adv.detach().requires_grad_(True)
                _, w = self.e4e_inversion_net(x_adv,randomize_noise=False, return_latents=True, resize=False,input_code=False) 

                masked_w = w * id_mask
                masked_w_opp = w_opp * id_mask

                loss = F.mse_loss(masked_w, masked_w_opp)
                loss.backward(retain_graph=True)
                
                x_grad = x_adv.grad.data
                norm = torch.norm(x_grad.view(x_grad.shape[0], -1), dim=1).view((-1, 1, 1, 1))
                norm = torch.clamp(norm, min=1e-6)
                x_grad /= norm

                grad = x_grad if grad is None else momentum * grad + x_grad
                x_adv = x_adv - alpha/ 255.0 * torch.sign(grad)
                x_adv = torch.max(x_adv, lower_bound)
                x_adv = torch.min(x_adv, upper_bound)
                x_adv.requires_grad_(True)

                global_config.training_step += 1
            
            perturbed_image = (x_adv.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()
            pillow_image = Image.fromarray(perturbed_image[0])
            wandb.log({f"output": [
                wandb.Image(pillow_image, caption=f"Output")]}, step=global_config.training_step)


            for param in self.G.parameters(): 
                param.requires_grad = True
            self.G.train()
