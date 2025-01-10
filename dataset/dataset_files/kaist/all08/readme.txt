原数据集：day（set00~02、set06~08），night（set03~05、set09~11）

v1.0（val和test中昼夜图像分配不均）
train：s00~s05，每8帧取一张图像
val：s06~s08，每10帧取一张图像
test：s09~s11，每10帧取一张图像

v2.0
train：s00~s05，每8帧取一张图像
val：s07、s08、s10，每10帧取一张图像
test：s06、s09、s11，每10帧取一张图像
