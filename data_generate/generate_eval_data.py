import wandb
import os

image_root = '/DDN_ROOT/wjpeng/dataset/FSC-147/images_384_VarV2'
ann_root = ''


def explore_fsc147():
    from IPython import embed
    embed()
    img_filenames = os.listdir(image_root)



if __name__ == '__main__':
    wandb.login()
    explore_fsc147()
