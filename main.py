from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 定义数据增强的字典
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')


myGene = trainGenerator(2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)

# 载入模型
model = unet()

# ModelCheckpoint:每一周期保存模型
# 1：保存模型的路径；2：monitor='loss'(检测loss，使其最小)；3：save_best_only=True(只保存在验证集上性能最优的模型)
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
# 训练生成器
# steps_per_epoch:指的是每个epoch有多少个batch size，其值=训练集总样本数 / batch size
model.fit_generator(myGene,steps_per_epoch=300,epochs=1,callbacks=[model_checkpoint])

#
testGene = testGenerator("data/membrane/test")

# 30为steps = 样本批次数
results = model.predict_generator(testGene,30,verbose=1)

# 保存结果
saveResult("data/membrane/test",results)