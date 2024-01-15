# DATA
data_root = './scripts/'
train_dataset = 'all_train_data.txt'
val_dataset = 'all_val_data.txt'
test_dataset = 'all_test_data.txt'

# TRAIN
epoch = 200
train_batch_size = 16
val_batch_size = 1
test_batch_size = 1
optimizer = 'SGD'  #['SGD','Adam']
learning_rate = 0.1
weight_decay = 1e-4
momentum = 0.9

scheduler = 'multi' #['multi', 'cos']
steps = [25, 38]
gamma  = 0.1
warmup = 'linear'
warmup_iters = 695

# NETWORK
use_aux = True
use_augment_in_train = True
use_augment_in_val = False
use_augment_in_test = False
num_x_grid = 200
num_y_grid = 18
backbone = '50'

# LOSS
sim_loss_w = 0.0
shp_loss_w = 0.0

# EXP
note = ''

log_path = 'logs'

# FINETUNE or RESUME MODEL PATH
finetune = None
resume = None

# TEST
test_model = None
test_work_dir = None

num_lanes = 12

# IMG SIZE
ori_img_width = 2560
ori_img_height = 1440

# resieze size
resized_width = 800
resized_height = 480

