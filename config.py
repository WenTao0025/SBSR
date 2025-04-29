import argparse

def config_argument():
    parser = argparse.ArgumentParser(description='This is for crossNet')
    # device
    parser.add_argument('--data_path',type=str,default='../data')
    parser.add_argument('--device',type=str,default='cuda:0')
    parser.add_argument('--channels',type=int,default=3)

    parser.add_argument('--batch_train_size',type=int,default=16)
    parser.add_argument('--lambda_pushaway',type=float,default=0.1)
    parser.add_argument('--num_views',type=int ,default=12)
    parser.add_argument('--embding_size',type=int,default=512)
    parser.add_argument('--class_num', type=int, default=171)
    # parser.add_argument('--class_num', type=int, default=90)
    # parser.add_argument('--class_num', type=int, default=48)
    parser.add_argument('--beta1',type=int,default=0.5)
    parser.add_argument('--beta2',type=int,default=0.999)
    parser.add_argument('--tau',type=float,default=0.5)

    #SHREC'13/SHREC'14
    # parser.add_argument('--maha_file',type=str,default='../params/VITB/mean_cov_PART_14.npy')
    # parser.add_argument('--maha_file', type=str, default='../params/VITB/mean_cov_13.npy')
    parser.add_argument('--maha_file', type=str, default='../params/VITB/mean_cov.npy')





    parser.add_argument('--left', default=1.0, type=float)
    parser.add_argument('--right', default=-1.0, type=float)
    #优化器
    parser.add_argument('--lr',type=float,default=1e-6)





    #数据集
    # parser.add_argument('--Sketch', type=str, default='RMD_Score_SHREC13_SBR_TRAINING_SKETCHES_New')
    # parser.add_argument('--Sketch',type=str,default='SHREC13_SBR_TRAINING_SKETCHES')
    parser.add_argument('--Sketch', type=str, default='SHREC14LSSTB_SKETCHES')
    # parser.add_argument('--Sketch', type=str, default='RMD_Score_SHREC14LSSTB_SKETCHES_New')
    # parser.add_argument('--Sketch', type=str, default="PART-SHREC'14")
    # parser.add_argument('--render',type=str,default=str("Shrec'13_render"))
    parser.add_argument('--render', type=str, default=str("render"))
    # parser.add_argument('--render', type=str, default=str("PART-SHREC'14-RENDER"))

    parser.add_argument('--shuffle',type=bool,default=True)
    parser.add_argument('--img_size',type=int,default=224)
    # parser.add_argument('--save_path',type=str,default='../params/VITB/new_params_cross_13.pth')
    parser.add_argument('--save_path',type=str,default='../params/VITB/new_params_cross_1401-26-16.pth')
    # parser.add_argument('--save_path',type=str,default='../params/VITB/params_cross_PART14.pth')




    parser.add_argument('--color_jitter', '-jitter', action='store_true')
    parser.add_argument('--LoG', '-LoG', action='store_true', help='use Laplacian of Gaussian for data augmentation')
    parser.add_argument('--grey', '-gr', action='store_true',
                        help='using gray scale images')




    #不确定性学习
    parser.add_argument('--e_lambda',type=int,default=0.3)
    parser.add_argument('-T',type=int,default=0.7)
    parser.add_argument('-c',type=float,default=1e-5)


    #单独训练模型域网络的参数
    parser.add_argument('--batch_train_shape_size', type=int, default=32)
    parser.add_argument('--model_train_lr',type=int,default=1e-5)
    parser.add_argument('--ModelNet_params',type=str,default='../params/VITB/backbone/ModelNet_13.pth')

    #单独训练草图域网络的参数
    parser.add_argument('--batch_train_sketch_size', type=int, default=64)
    parser.add_argument('--sketch_train_lr',type = int,default=1e-6)
    parser.add_argument('--SketchNet_params',type=str,default='../params/VITB/backbone/QuerySketchNet_13.pth')

    arg = parser.parse_args()
    return arg

