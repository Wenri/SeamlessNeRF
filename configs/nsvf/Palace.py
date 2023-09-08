_base_ = '../default.py'

expname = 'dvgo_Palace'
basedir = './logs/nsvf_synthetic'
target = 'dvgo_Toad'
data = dict(
    datadir='./data/Synthetic_NSVF/Palace',
    dataset_type='nsvf',
    inverse_y=True,
    white_bkgd=True,
)

matrix = (0.7702, 0.0000, 0.0000, 0.1255, 0.0000, 0.7702, 0.0000, -0.0732, 0.0000, 0.0000, 0.7702, -0.1245, 0.0000,
          0.0000, 0.0000, 1.0000)
