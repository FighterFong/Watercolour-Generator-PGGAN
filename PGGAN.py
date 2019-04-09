# encoding=utf-8
import tensorflow as tf
from ops import lrelu, conv2d, fully_connect, upscale, Pixl_Norm, downscale2d, MinibatchstateConcat
from utils import save_images, check_path
import numpy as np
from scipy.ndimage.interpolation import zoom
import logging  # 引入logging模块
import os
import time
import tqdm

class PGGAN(object):

    # build model
    def __init__(self, batch_size, max_iters, model_path, read_model_path, data, sample_size, sample_path, log_dir,
                 learn_rate, lam_gp, lam_eps, PG, trans, use_wscale, is_celeba, step_by_save_sample, step_by_save_weights):
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.gan_model_path = model_path
        self.read_model_path = read_model_path
        self.data_In = data
        self.sample_size = sample_size
        self.sample_path = sample_path
        self.log_dir = log_dir
        self.learning_rate = learn_rate
        self.lam_gp = lam_gp
        self.lam_eps = lam_eps
        self.pg = PG
        self.trans = trans
        self.log_vars = []
        self.channel = self.data_In.channel
        self.output_size = 4 * pow(2, PG - 1)
        self.use_wscale = use_wscale
        self.is_celeba = is_celeba
        self.images = tf.placeholder(tf.float32, [batch_size, self.output_size, self.output_size, self.channel])
        # laten vector
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.sample_size])
        self.alpha_tra = tf.Variable(initial_value=0.0, trainable=False, name='alpha_tra')
        self.step_by_save_sample = step_by_save_sample
        self.step_by_save_weights = step_by_save_weights
        self.init_logger()

    # def init_logger(self):
    #     # 第一步，创建一个logger
    #     self.logger = logging.getLogger()
    #     self.logger.setLevel(logging.INFO)  # Log等级总开关
    #     # 第二步，创建一个handler，用于写入日志文件
    #     rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    #     # log_path = os.path.dirname(os.getcwd()) + '/Logs/'
    #     log_path = os.path.join(os.getcwd(), 'Logs')
    #     check_path(log_path)
    #     log_name = os.path.join(log_path, rq + '.log')
    #     logfile = log_name
    #     fh = logging.FileHandler(logfile, mode='w')
    #     fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    #     # 第三步，定义handler的输出格式
    #     formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    #     fh.setFormatter(formatter)
    #     # 第四步，将logger添加到handler里面
    #     self.logger.addHandler(fh)

    def init_logger(self):
        self.f_logger = open('Logs/' + str(self.pg) + '.log', 'a')

    def build_model_PGGan(self):
        self.fake_images = self.generate(self.z, pg=self.pg, t=self.trans, alpha_trans=self.alpha_tra)
        # 真、假图片判断为真实图片的概率
        _, self.D_pro_logits = self.discriminate(self.images, reuse=False, pg = self.pg, t=self.trans, alpha_trans=self.alpha_tra)
        _, self.G_pro_logits = self.discriminate(self.fake_images, reuse=True, pg = self.pg, t=self.trans, alpha_trans=self.alpha_tra)

        # 定义loss。the defination of loss for D and G
        self.D_loss = tf.reduce_mean(self.G_pro_logits) - tf.reduce_mean(self.D_pro_logits)
        self.G_loss = -tf.reduce_mean(self.G_pro_logits)

        # gradient penalty from WGAN-GP
        self.differences = self.fake_images - self.images
        self.alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        interpolates = self.images + (self.alpha * self.differences)
        _, discri_logits= self.discriminate(interpolates, reuse=True, pg=self.pg, t=self.trans, alpha_trans=self.alpha_tra)
        gradients = tf.gradients(discri_logits, [interpolates])[0]

        # 2 norm
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
        self.gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        tf.summary.scalar("gp_loss", self.gradient_penalty)

        self.D_origin_loss = self.D_loss
        self.D_loss += self.lam_gp * self.gradient_penalty
        self.D_loss += self.lam_eps * tf.reduce_mean(tf.square(self.D_pro_logits))

        self.log_vars.append(("generator_loss", self.G_loss))
        self.log_vars.append(("discriminator_loss", self.D_loss))

        # all the variables
        t_vars = tf.trainable_variables()
        # all the discriminator variables
        self.d_vars = [var for var in t_vars if 'dis' in var.name]

        ## Print Generator And Discriminator's Architecture and parameter number
        total_para = 0
        for variable in self.d_vars:
            shape = variable.get_shape()
            # print (variable.name, shape)
            # self.logger.debug(variable.name, shape)
            self.f_logger.write(variable.name + ' ' + str(shape) + '\n')
            variable_para = 1
            for dim in shape:
                variable_para *= dim.value
            total_para += variable_para
        # print ("The total para of D", total_para)
        # self.logger.debug("The total para of D", total_para)
        self.f_logger.write("The total para of D" + ' ' + str(total_para) + '\n')

        # all the generator variables
        self.g_vars = [var for var in t_vars if 'gen' in var.name]

        total_para2 = 0
        for variable in self.g_vars:
            shape = variable.get_shape()
            # print (variable.name, shape)
            # self.logger.debug(variable.name, shape)
            self.f_logger.write(variable.name + ' ' + str(shape) + '\n')
            variable_para = 1
            for dim in shape:
                variable_para *= dim.value
            total_para2 += variable_para
        # print ("The total para of G", total_para2)
        # self.logger.debug("The total para of G", total_para2)
        self.f_logger.write("The total para of G" + ' ' + str(total_para2) + '\n')

        # save the variables , which remain unchanged
        self.d_vars_n = [var for var in self.d_vars if 'dis_n' in var.name]
        self.g_vars_n = [var for var in self.g_vars if 'gen_n' in var.name]

        # remove the new variables for the new model
        self.d_vars_n_read = [var for var in self.d_vars_n if '{}'.format(self.output_size) not in var.name]
        self.g_vars_n_read = [var for var in self.g_vars_n if '{}'.format(self.output_size) not in var.name]

        # save the rgb variables, which remain unchanged
        self.d_vars_n_2 = [var for var in self.d_vars if 'dis_y_rgb_conv' in var.name]
        self.g_vars_n_2 = [var for var in self.g_vars if 'gen_y_rgb_conv' in var.name]

        self.d_vars_n_2_rgb = [var for var in self.d_vars_n_2 if '{}'.format(self.output_size) not in var.name]
        self.g_vars_n_2_rgb = [var for var in self.g_vars_n_2 if '{}'.format(self.output_size) not in var.name]

        # print ("d_vars", len(self.d_vars))
        # print ("g_vars", len(self.g_vars))

        # print ("self.d_vars_n_read", len(self.d_vars_n_read))
        # print ("self.g_vars_n_read", len(self.g_vars_n_read))

        # print ("d_vars_n_2_rgb", len(self.d_vars_n_2_rgb))
        # print ("g_vars_n_2_rgb", len(self.g_vars_n_2_rgb))

        # for n in self.d_vars:
        #     print (n.name)

        self.g_d_w = [var for var in self.d_vars + self.g_vars if 'bias' not in var.name]

        # print ("self.g_d_w", len(self.g_d_w))

        ## Saver Define
        # saver 用来存储网络参数，将discriminator和generator所有参数都存储起来
        self.saver = tf.train.Saver(self.d_vars + self.g_vars)
        # 只读取部分参数
        self.r_saver = tf.train.Saver(self.d_vars_n_read + self.g_vars_n_read)
        if len(self.d_vars_n_2_rgb + self.g_vars_n_2_rgb):
            self.rgb_saver = tf.train.Saver(self.d_vars_n_2_rgb + self.g_vars_n_2_rgb)

        for k, v in self.log_vars:
            tf.summary.scalar(k, v)

    # do train
    def train(self):
        step_pl = tf.placeholder(tf.float32, shape=None)
        # this what？seems modify alpha_tra by step
        alpha_tra_assign = self.alpha_tra.assign(step_pl / self.max_iters)

        # 定义优化函数
        opti_D = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.0, beta2=0.99).minimize(
            self.D_loss, var_list=self.d_vars)
        opti_G = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.0, beta2=0.99).minimize(
            self.G_loss, var_list=self.g_vars)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(init)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)
            # print('== [INFO] PG:', self.pg)
            # self.logger.debug('== [INFO] PG:', self.pg)
            self.f_logger.write('== [INFO] PG:' + ' ' + str(self.pg) + '\n')
            # 除了1和7，其他都应该从上一层的计算中restore上一层的参数
            # if self.pg != 1 and self.pg != 7: # self.pg = 1 相当于循环第一次，即第0次，第0次不载入预训练weight， self.pg != 7不会等于7
            if self.pg != 1: # self.pg = 1 相当于循环第一次，即第0次， self.pg != 7不会等于7
                if self.trans: # 外层总共循环11次，偶数次才会进来，这个目的是用于Training
                    self.r_saver.restore(sess, self.read_model_path)
                    self.rgb_saver.restore(sess, self.read_model_path)
                    # print('== [INFO] Restore Checkpoint Complete! ==')
                    # self.logger.debug('== [INFO] Restore RGB Checkpoint Complete! ==')
                    self.f_logger.write('== [INFO] Restore RGB Checkpoint Complete! ==' + '\n')
                else: # 用于Testing，输出测试效果
                    self.saver.restore(sess, self.read_model_path)
                    # print('== [INFO] Restore All variable Complete! ==')
                    # self.logger.debug('== [INFO] Restore All variable Complete! ==')
                    self.f_logger.write('== [INFO] Restore All variable Complete! ==' + '\n')
            # 第一层计算，如果有预训练参数，则加载
            elif self.pg == 1:
                pass

            step = 0
            batch_num = 0
            # circle training
            # while step <= self.max_iters:
            for step in tqdm.tqdm(range(self.max_iters)):
                # optimization D，不知道n_critic意义何在
                n_critic = 1
                if self.pg >= 5:
                    n_critic = 1

                try:
                    # 不知道这个for循环的意义何在
                    for i in range(n_critic):
                        # Define latent vector
                        sample_z = np.random.normal(size=[self.batch_size, self.sample_size])
                        # Load data
                        if self.is_celeba:
                            train_list = self.data_In.getNextBatch(batch_num, self.batch_size)
                            realbatch_array = self.data_In.getShapeForData(train_list, resize_w=self.output_size)
                        else:
                            realbatch_array = self.data_In.getNextBatch(self.batch_size, resize_w=self.output_size)
                            realbatch_array = np.transpose(realbatch_array, axes=[0, 3, 2, 1]).transpose([0, 2, 1, 3])

                        if self.trans and self.pg != 0: # self.trans：training或者testing self.pg 不会等于0
                            alpha = np.float(step) / self.max_iters
                            low_realbatch_array = zoom(realbatch_array, zoom=[1, 0.5, 0.5, 1], mode='nearest')
                            low_realbatch_array = zoom(low_realbatch_array, zoom=[1, 2, 2, 1], mode='nearest')
                            # 按一定的alpha，将图片转换成高分辨率 + 低分辨率的线性组合，起到加噪声的作用？？
                            realbatch_array = alpha * realbatch_array + (1 - alpha) * low_realbatch_array

                        # 训练判别器。training discriminator first
                        sess.run(opti_D, feed_dict={self.images: realbatch_array, self.z: sample_z})
                        batch_num += 1
                # Ensure program running once Datasets format has error.
                except Exception as e:
                    batch_num += 1
                    continue

                # next step to training optimization Generator
                sess.run(opti_G, feed_dict={self.z: sample_z})

                summary_str = sess.run(summary_op, feed_dict={self.images: realbatch_array, self.z: sample_z})
                summary_writer.add_summary(summary_str, step)
                summary_writer.add_summary(summary_str, step)
                # the alpha of fake_in process
                sess.run(alpha_tra_assign, feed_dict={step_pl: step})

                if step % self.step_by_save_sample == 0:
                    D_loss, G_loss, D_origin_loss, alpha_tra = sess.run([self.D_loss, self.G_loss, self.D_origin_loss,self.alpha_tra], feed_dict={self.images: realbatch_array, self.z: sample_z})
                    # print("PG %d, step %d: D loss=%.7f G loss=%.7f, D_or loss=%.7f, opt_alpha_tra=%.7f" % (self.pg, step, D_loss, G_loss, D_origin_loss, alpha_tra))
                    # self.logger.debug("PG %d, step %d: D loss=%.7f G loss=%.7f, D_or loss=%.7f, opt_alpha_tra=%.7f" % (self.pg, step, D_loss, G_loss, D_origin_loss, alpha_tra))
                    self.f_logger.write("PG %d, step %d: D loss=%.7f G loss=%.7f, D_or loss=%.7f, opt_alpha_tra=%.7f" % (self.pg, step, D_loss, G_loss, D_origin_loss, alpha_tra) + '\n')

                    # normalize and save real images
                    realbatch_array = np.clip(realbatch_array, -1, 1)
                    # print('1-',type(realbatch_array[0:self.batch_size]))
                    # print('1-',realbatch_array[0:self.batch_size].shape)
                    # 用来指定将一个batch的图片制作成一个大图的尺寸，2行，(batch/2)列 
                    save_images(realbatch_array[0:self.batch_size], [2, self.batch_size/2], '{}/{:02d}_real.jpg'.format(self.sample_path, step)) 

                    # 存储低分辨率真实图片
                    if self.trans and self.pg != 0:
                        low_realbatch_array = np.clip(low_realbatch_array, -1, 1)
                        save_images(low_realbatch_array[0:self.batch_size], [2, self.batch_size / 2],
                                    '{}/{:02d}_real_lower.jpg'.format(self.sample_path, step))
                   
                    # 生成图片。generate fake image
                    fake_image = sess.run(self.fake_images,
                                          feed_dict={self.images: realbatch_array, self.z: sample_z})
                    fake_image = np.clip(fake_image, -1, 1)
                    save_images(fake_image[0:self.batch_size], [2, self.batch_size/2], '{}/{:02d}_train.jpg'.format(self.sample_path, step))

                # 存储 weights
                if np.mod(step, self.step_by_save_weights) == 0 and step != 0:
                    # 只用到了 saver 这个来存储
                    self.saver.save(sess, self.gan_model_path)
                    # ./output/Experiment_6_30_1/model_pggan_0/1/
                    # print("Model saved in file: %s" % self.gan_model_path)
                    # self.logger.debug("Model saved in file: %s" % self.gan_model_path)
                    self.f_logger.write("Model saved in file: %s" % self.gan_model_path + '\n')

                step += 1

            save_path = self.saver.save(sess, self.gan_model_path)
            # print("Model saved in file: %s" % save_path)
            # self.logger.debug("Model saved in file: %s" % save_path)
            self.f_logger.write("Model saved in file: %s" % save_path + '\n')

        tf.reset_default_graph()

    def discriminate(self, conv, reuse=False, pg=1, t=False, alpha_trans=0.01):
        #dis_as_v = []
        with tf.variable_scope("discriminator") as scope:

            if reuse == True:
                scope.reuse_variables()
            # t 代表是否进行training，如果进行training则进行下述操作
            # 
            if t:
                conv_iden = downscale2d(conv)
                #from RGB
                conv_iden = lrelu(conv2d(conv_iden, output_dim= self.get_nf(pg - 2), k_w=1, k_h=1, d_h=1, d_w=1, use_wscale=self.use_wscale,
                           name='dis_y_rgb_conv_{}'.format(conv_iden.shape[1])))
            # fromRGB
            conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 1), k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, name='dis_y_rgb_conv_{}'.format(conv.shape[1])))
            # pg 代表当前 pixel 上升至哪个档位
            # pg=1，即第一层的training不需要进入，不需要提升pixel并进行一系列操作
            for i in range(pg - 1):
                conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 1 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,
                                    name='dis_n_conv_1_{}'.format(conv.shape[1])))
                conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,
                                                      name='dis_n_conv_2_{}'.format(conv.shape[1])))
                conv = downscale2d(conv)
                if i == 0 and t:
                    conv = alpha_trans * conv + (1 - alpha_trans) * conv_iden

            conv = MinibatchstateConcat(conv)
            conv = lrelu(
                conv2d(conv, output_dim=self.get_nf(1), k_w=3, k_h=3, d_h=1, d_w=1, use_wscale=self.use_wscale, name='dis_n_conv_1_{}'.format(conv.shape[1])))
            conv = lrelu(
                conv2d(conv, output_dim=self.get_nf(1), k_w=4, k_h=4, d_h=1, d_w=1, use_wscale=self.use_wscale, padding='VALID', name='dis_n_conv_2_{}'.format(conv.shape[1])))
            conv = tf.reshape(conv, [self.batch_size, -1])

            # for D
            output = fully_connect(conv, output_size=1, use_wscale=self.use_wscale, gain=1, name='dis_n_fully')

            return tf.nn.sigmoid(output), output

    def generate(self, z_var, pg=1, t=False, alpha_trans=0.0):

        with tf.variable_scope('generator') as scope:
            # 
            de = tf.reshape(Pixl_Norm(z_var), [self.batch_size, 1, 1, int(self.get_nf(1))])
            de = conv2d(de, output_dim=self.get_nf(1), k_h=4, k_w=4, d_w=1, d_h=1, use_wscale=self.use_wscale, gain=np.sqrt(2)/4, padding='Other', name='gen_n_1_conv')
            de = Pixl_Norm(lrelu(de))
            de = tf.reshape(de, [self.batch_size, 4, 4, int(self.get_nf(1))])
            de = conv2d(de, output_dim=self.get_nf(1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_2_conv')
            de = Pixl_Norm(lrelu(de))

            # pg 代表当前 pixel 上升至哪个档位
            # pg=1，即第一层的training不需要进入，不需要提升pixel并进行一系列操作
            for i in range(pg - 1):
                # t 代表是否进行training，如果进行training则进行下述操作
                # 
                if i == pg - 2 and t:
                    # To RGB
                    de_iden = conv2d(de, output_dim=3, k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale,
                                     name='gen_y_rgb_conv_{}'.format(de.shape[1]))
                    de_iden = upscale(de_iden, 2)

                de = upscale(de, 2)
                de = Pixl_Norm(lrelu(
                    conv2d(de, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_conv_1_{}'.format(de.shape[1]))))
                de = Pixl_Norm(lrelu(
                    conv2d(de, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_conv_2_{}'.format(de.shape[1]))))

            # To RGB
            de = conv2d(de, output_dim=3, k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, gain=1, name='gen_y_rgb_conv_{}'.format(de.shape[1]))

            if pg == 1: return de
            # 如果是training，通过 alpha_trans 来调控和线性组合图片
            if t: 
                de = (1 - alpha_trans) * de_iden + alpha_trans*de
            else: 
                de = de

            return de

    def get_nf(self, stage):
        return min(1024 / (2 **(stage * 1)), 512)
    
    def sample_z(self, mu, log_var):
        eps = tf.random_normal(shape=tf.shape(mu))
        return mu + tf.exp(log_var / 2) * eps










