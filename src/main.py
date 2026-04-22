import argparse
import os
from pathlib import Path

from src.config.config import Config
from src.exp.base_exp import BaseExp
from src.utils.utils import read_cfg, logging_init, fix_random_seed


def parse_args():
    """
    解析命令行、配置文件，构建配置类
    """
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='Deep learning template project')
    # 配置文件
    parser.add_argument('-cfg', default='./configs/base.yaml',
                        metavar='<path/to/cfg>', help='path to config file')
    # 可选参数
    parser.add_argument('-train', dest='train_model', action='store_true', default=False,
                        help='train model')
    parser.add_argument('-val', dest='val_model', action='store_true', default=False,
                        help='validate model')
    parser.add_argument('-test', dest='test_model', action='store_true', default=False,
                        help='test model')
    parser.add_argument('-resume', metavar='<ckpt_filename>',
                        help='resume from checkpoint')
    parser.add_argument('-once', action='store_true', default=False,
                        help='test train model structure once')
    # 在此添加更多命令行参数

    # 解析参数
    args = parser.parse_args()

    # 设置工作目录为项目根目录
    args.proj_root = os.getcwd()
    # 设置配置文件夹名_数据集为实验名
    args.exp_name = Path(args.cfg).parent.name + "_" + Path(args.cfg).stem
    # 设置配置文件夹名为模型名 for 对比实验
    args.model_name = Path(args.cfg).parent.name
    # 设置实验输出目录
    args.output_path = Path(args.proj_root, 'outputs', args.exp_name)

    # 读取配置文件
    base_cfg = read_cfg(Path(args.cfg).parent / 'base.yaml')
    data_cfg = read_cfg(args.cfg)

    # 设置所有参数全局可用
    Config.update_args([vars(args), base_cfg, data_cfg])


def main():
    """
    项目主函数
    """
    # 解析参数
    parse_args()

    # 配置logging
    logging_init()

    # 整齐打印参数，必须在配置logging之后
    Config.logging_args()

    # 固定随机种子
    fix_random_seed(seed=Config.args.seed)

    # 数据加载
    # train_loader：训练集
    # val_loader：测试集（与训练集不交叉），指标可信
    # test_loader：所有数据，用于可视化，指标不可信
    # 创建实验实例
    exp = BaseExp()

    if Config.args.once:
        exp.once()

    # 模型训练
    if Config.args.train_model:
        exp.train()

    # 模型验证
    if Config.args.val_model:
        exp.validate()

    # 模型测试
    if Config.args.test_model:
        exp.test()


if __name__ == '__main__':
    main()
