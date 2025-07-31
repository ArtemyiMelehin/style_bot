"""
За основу был взят код pytorch-CycleGAN-and-pix2pix
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
Некоторые функции и настройки запуска были изменены так,
чтобы сеть принимала изображение,
обрабатывала его генератором и выдавала результат.

Предварительно скачаны веса предобученных моделей
style_vangogh.pth  и winter2summer_yosemite.pth из хранилища
http://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/
в
transform_models/cycleGAN/checkpoints/

Подготовка весов для работы бота производится с помощью
данного вспомогательного скрипта.

Нужно указать имена файлов скачанных моделей

Подготовленные для использования ботом веса сохраняются скриптом в
transform_models/pth/
"""


from options.test_options import TestOptions
from models import create_model
import torch
from shutil import copy2
import os

# Источник:
# http://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/

pretrain_model_filename = "style_vangogh.pth"
path_result_model = "../pth/vangogh.pth"

# pretrain_model_filename = "winter2summer_yosemite.pth"
# path_result_model = "../pth/winter2summer.pth"


if __name__ == "__main__":
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.serial_batches = True
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    opt.model = 'test'
    opt.dataroot = '/pth/'
    # opt.checkpoints_dir = '/pth/'
    opt.continue_train = False
    opt.name = ''
    opt.no_dropout = True
    
    print(os.listdir(os.path.join(opt.checkpoints_dir)))
    
    pretrain_model_filename = os.path.join(opt.checkpoints_dir, pretrain_model_filename)
    checkpoint_filename = os.path.join(opt.checkpoints_dir, 'latest_net_G.pth')
    
    print(pretrain_model_filename)
    
    copy2(pretrain_model_filename, checkpoint_filename)
    
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)
    if opt.eval:
        model.eval()
    
    torch.save(model, path_result_model)
