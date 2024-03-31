import argparse
import cv2
import glob
import os
import shutil
import torch

from basicsr.archs.basicvsr_x8_arch import BasicVSR_x8
from basicsr.data.data_util import read_img_seq
from basicsr.utils.img_util import tensor2img

from pyinstrument import Profiler


def inference(imgs, imgnames, model, save_path):
    with torch.no_grad():
        outputs = model(imgs)
    # save imgs
    outputs = outputs.squeeze()
    outputs = list(outputs)
    for output, imgname in zip(outputs, imgnames):
        #最费时间的地方，需要优化
        output = tensor2img(output)
        cv2.imwrite(os.path.join(save_path, f'{imgname}_BasicVSR.png'), output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='experiments/pretrained_models/BasicVSR_REDS4_X8.pth')
    parser.add_argument(
        '--input_path', type=str, default='datasets/REDS4/sharp_bicubic/000', help='input test image folder')
    parser.add_argument('--save_path', type=str, default='results/BasicVSR_x8', help='save image path')
    parser.add_argument('--interval', type=int, default=15, help='interval size')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    profiler.start()

    # set up model
    print('set up model')
    model = BasicVSR_x8(num_feat=64, num_block=30)
    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)

    os.makedirs(args.save_path, exist_ok=True)

    # extract images from video format files
    print('loading video')
    input_path = args.input_path
    use_ffmpeg = False
    if not os.path.isdir(input_path):
        use_ffmpeg = True
        video_name = os.path.splitext(os.path.split(args.input_path)[-1])[0]
        input_path = os.path.join('./BasicVSR_tmp', video_name)
        os.makedirs(os.path.join('./BasicVSR_tmp', video_name), exist_ok=True)
        os.system(f'ffmpeg -i {args.input_path} -qscale:v 1 -qmin 1 -qmax 1 -vsync 0  {input_path} /frame%08d.png')

    # load data and inference
    imgs_list = sorted(glob.glob(os.path.join(input_path, '*')))
    num_imgs = len(imgs_list)
    print('load data and inference')
    if len(imgs_list) <= args.interval:  # too many images may cause CUDA out of memory
        imgs, imgnames = read_img_seq(imgs_list, return_imgname=True)
        imgs = imgs.unsqueeze(0).to(device)
        inference(imgs, imgnames, model, args.save_path)
    else:
        for idx in range(0, num_imgs, args.interval):
            print('processing frame',idx+1)
            interval = min(args.interval, num_imgs - idx)
            imgs, imgnames = read_img_seq(imgs_list[idx:idx + interval], return_imgname=True)
            imgs = imgs.unsqueeze(0).to(device)
            inference(imgs, imgnames, model, args.save_path)

    # delete ffmpeg output images
    if use_ffmpeg:
        print('cleaning')
        shutil.rmtree(input_path)

    profiler.stop()
    profiler.output_html()
    profiler.write_html('basicvsr_x8.html')


if __name__ == '__main__':
    profiler = Profiler()
    main()
