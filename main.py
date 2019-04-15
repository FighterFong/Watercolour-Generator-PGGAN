# encoding=utf-8
import tensorflow as tf

from utils import mkdir_p
from PGGAN import PGGAN
from utils import CelebA, CelebA_HQ, print_msg
flags = tf.app.flags
import os
import datetime

os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'

flags.DEFINE_string("OPER_NAME", "Test", "the name of experiments")  #  Calligraphy
flags.DEFINE_integer("OPER_FLAG", 0, "Flag of opertion: 0 is for training ")
flags.DEFINE_string("path" , '?', "Path of training data, for example /home/hehe/")
flags.DEFINE_integer("batch_size", 16, "Batch size") # 32
flags.DEFINE_integer("sample_size", 512, "Size of sample")
flags.DEFINE_integer("max_iters", 300, "Maxmization of training number") # 40000
flags.DEFINE_float("learn_rate", 0.001, "Learning rate for G and D networks")
flags.DEFINE_integer("lam_gp", 10, "Weight of gradient penalty term")
flags.DEFINE_float("lam_eps", 0.001, "Weight for the epsilon term")
flags.DEFINE_integer("flag", 7, "FLAG of gan training process")
flags.DEFINE_boolean("use_wscale", True, "Using the scale of weight")
flags.DEFINE_boolean("celeba", True, "Whether using celeba or using CelebA-HQ")
flags.DEFINE_integer("step_by_save_sample", 50, "FLAG of gan training process") # 5000
flags.DEFINE_integer("step_by_save_weights", 250, "FLAG of gan training process") # 20000

FLAGS = flags.FLAGS
if __name__ == "__main__":

    root_log_dir = "./output/{}/logs/".format(FLAGS.OPER_NAME)
    mkdir_p(root_log_dir)

    if FLAGS.celeba:
        print_msg('INFO', 'Load data from {}.'.format(FLAGS.path))
        data_In = CelebA(FLAGS.path)
    else:
        print_msg('INFO', 'Load data from {}.'.format(FLAGS.path))
        data_In = CelebA_HQ(FLAGS.path)
    print("== The Number Of Dataset:", len(data_In.image_list))

    if FLAGS.OPER_FLAG == 0:

        # 相同数字代表：训练一回合，测试下一回合
        # fl = [1,2,2,3,3,4,4,5,5,6,6,7,7,8,8]
        # for jellyfish 
        fl = [1,2,2,3,3,4,4,5,5,6,6]
        # for jellyfish 4w
        # fl = [1,2,2]
        # for tinymind calligraphy
        # fl = [1,2,2,3,3,4,4]
        # for hkust sea view
        # fl = [1,2,2,3,3,4,4,5,5,6,6,7,7]

        # 读取上一轮 training 保存的参数，数字错开的，确保fl循环的时候能加载上一循环的参数
        # r_fl = [1,1,2,2,3,3,4,4,5,5,6,6,7,7,8]
        # for jellyfish 
        r_fl = [1,1,2,2,3,3,4,4,5,5,6]
        # for jellyfish 4w
        # r_fl = [1,1,2]
        # for tinymind calligraphy
        # r_fl = [1,1,2,2,3,3,4]
        # for hkust sea view
        # r_fl = [1,1,2,2,3,3,4,4,5,5,6,6,7]

        for i in range(FLAGS.flag):
            # 为什么把原来的true和false对换，trainset和生成的图片都会出现颜色偏黄的怪异现象
            # 偶数次（0 2 4 6 8）是false，奇数次（1 3 5 7）是true
            t = False if (i % 2 == 0) else True # 第1轮不用train是因为没有可以加载的weight,同时不用制作低分辨率原图+原图和保存低分辨率图
            # t = False if (i+1 % 2 == 0) else True
            pggan_checkpoint_dir_write = "./output/{}/model_pggan_{}/{}/".format(FLAGS.OPER_NAME, FLAGS.OPER_FLAG, fl[i])
            # sample_x_ture,sample_x_dalse,
            sample_path = "./output/{}/{}/sample_{}_{}".format(FLAGS.OPER_NAME, FLAGS.OPER_FLAG, fl[i], t)
            mkdir_p(pggan_checkpoint_dir_write)
            mkdir_p(sample_path)
            pggan_checkpoint_dir_read = "./output/{}/model_pggan_{}/{}/".format(FLAGS.OPER_NAME, FLAGS.OPER_FLAG, r_fl[i])

            pggan = PGGAN(
                            oper_name=FLAGS.OPER_NAME,batch_size=FLAGS.batch_size, max_iters=FLAGS.max_iters,
                            model_path=pggan_checkpoint_dir_write, read_model_path=pggan_checkpoint_dir_read,
                            data=data_In, sample_size=FLAGS.sample_size,
                            sample_path=sample_path, log_dir=root_log_dir, learn_rate=FLAGS.learn_rate, lam_gp=FLAGS.lam_gp, lam_eps=FLAGS.lam_eps, PG= fl[i],
                            trans=t, use_wscale=FLAGS.use_wscale, is_celeba=FLAGS.celeba,
                            step_by_save_sample=FLAGS.step_by_save_sample, step_by_save_weights=FLAGS.step_by_save_weights
                        )

            pggan.build_model_PGGan()

            start_time = datetime.datetime.now()
            pggan.train()
            end_time = datetime.datetime.now()
            pggan.f_logger.write('start_time:{}'.format(str(start_time)) + '\n')
            pggan.f_logger.write('end_time:{}'.format(str(end_time)) + '\n')
            pggan.f_logger.write('=='*20 + '\n')
            pggan.f_logger.write('=='*20 + '\n')
            pggan.f_logger.close()

    else:
        # 相同数字代表：训练一回合，下一回合进行测试
        fl = [1,2,2,3,3,4,4,5,5,6,6,7]
        # 错开，确保fl循环的时候能加载上一循环的参数
        r_fl = [1,1,2,2,3,3,4,4,5,5,6,6]

        i = 8
        t = False # True False
        pggan_checkpoint_dir_write = "./output/{}/model_pggan_{}/{}/".format(FLAGS.OPER_NAME, 0, fl[i])
        # sample_x_ture, sample_x_dalse,
        sample_path = "./output/{}/{}/sample_{}_{}".format(FLAGS.OPER_NAME, FLAGS.OPER_FLAG, fl[i], t)
        mkdir_p(pggan_checkpoint_dir_write)
        mkdir_p(sample_path)
        pggan_checkpoint_dir_read = "./output/{}/model_pggan_{}/{}/".format(FLAGS.OPER_NAME, 0, r_fl[i])

        pggan = PGGAN(
                        batch_size=FLAGS.batch_size, max_iters=FLAGS.max_iters,
                        model_path=pggan_checkpoint_dir_write, read_model_path=pggan_checkpoint_dir_read,
                        data=data_In, sample_size=FLAGS.sample_size,
                        sample_path=sample_path, log_dir=root_log_dir, learn_rate=FLAGS.learn_rate, lam_gp=FLAGS.lam_gp, lam_eps=FLAGS.lam_eps, PG= fl[i],
                        trans=t, use_wscale=FLAGS.use_wscale, is_celeba=FLAGS.celeba,
                        step_by_save_sample=FLAGS.step_by_save_sample, step_by_save_weights=FLAGS.step_by_save_weights
                    )

        pggan.build_model_PGGan()

        start_time = datetime.datetime.now()
        pggan.train()
        end_time = datetime.datetime.now()
        pggan.f_logger.write('start_time:{}'.format(str(start_time)) + '\n')
        pggan.f_logger.write('end_time:{}'.format(str(end_time)) + '\n')
        pggan.f_logger.write('=='*20 + '\n')
        pggan.f_logger.write('=='*20 + '\n')
        pggan.f_logger.close()













