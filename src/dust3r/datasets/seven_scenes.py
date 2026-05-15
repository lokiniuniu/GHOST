# src/dust3r/datasets/seven_scenes.py

import sys
from pathlib import Path
import torch.utils.data as data

sys.path.append(str(Path(__file__).parent.parent / 'eval/mv_recon'))
from data import SevenScenes as SevenScenesEvalDataset

class SevenScenesDataset(data.Dataset): # <-- 直接继承自 PyTorch 的 Dataset 类
    def __init__(self, root, split='train', **kwargs):
        super().__init__()
        assert split in ['train', 'test'], "Split must be 'train' or 'test'"
        
        self.root = Path(root)
        self.split = split
        self.kwargs = kwargs # 保存其他参数以传递给加载器
        self.scenes = self._get_scenes()
        
        if not self.scenes:
            raise FileNotFoundError(
                f"No scenes found. Please check that '{root}' is the correct 7-Scenes root directory "
                f"and contains subdirectories like 'chess', 'fire', etc., each with a '{split.capitalize()}Split.txt' file."
            )

    def _get_scenes(self):
        """
        根据官方的 TrainSplit.txt 或 TestSplit.txt 文件来构建场景列表。
        """
        scenes_list = []
        # 遍历根目录下的所有子目录 (e.g., 'chess', 'fire', 'heads')
        for scene_dir in self.root.iterdir():
            if not scene_dir.is_dir():
                continue

            split_file = scene_dir / f'{self.split.capitalize()}Split.txt'
            
            if split_file.exists():
                with open(split_file, 'r') as f:
                    # 读取文件中的每一行，例如 "sequence1", "sequence2"
                    sequences = [line.strip() for line in f.readlines()]
                    for seq in sequences:
                        # 将 "sequence1" 转换为 "seq-01"
                        seq_name = f"seq-{int(seq.replace('sequence', '')):02d}"
                        # 构建完整的场景标识符，例如 "chess/seq-01"
                        full_scene_path = f"{scene_dir.name}/{seq_name}"
                        scenes_list.append(full_scene_path)
        return scenes_list

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene_path = self.scenes[idx]
        
        # 使用原始的加载器来加载单个指定的场景
        # 我们通过创建一个临时的 SevenScenesEvalDataset 实例来实现
        # 注意：这里的 'scenes' 参数接收一个列表
        item_loader = SevenScenesEvalDataset(self.root, scenes=[scene_path], **self.kwargs)
        
        # 加载并返回这个场景的数据
        return item_loader[0]