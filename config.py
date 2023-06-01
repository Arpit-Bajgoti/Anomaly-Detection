import torch

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"
TRAIN_DIR = "Train"
TEST_DIR = "Test"
LEARNING_RATE = 2e-4
BATCH_SIZE = 8
NUM_WORKERS = 8
IMAGE_SIZE = 224
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 500
LOAD_MODEL = False
SAVE_MODEL = False
# CHECKPOINT_DISC = "trained_checkpoints_binary_gen/disc.pth.tar"
CHECKPOINT_GEN = "D:\gen.pth.tar"
# CHECKPOINT_GEN = "D:\gen.pth.tar"
