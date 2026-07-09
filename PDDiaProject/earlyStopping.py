import torch


class EarlyStopping:
    """
    基于AUC的早停止

    当验证集AUC连续 patience 轮不提升时停止训练
    """

    def __init__(
        self,
        patience=50,
        min_delta=1e-4,
        save_path="best_model.pth"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path

        self.best_auc = -1

        self.counter = 0

        self.early_stop = False

    def __call__(
        self,
        auc,
        model
    ):

        # 第一次记录
        if self.best_auc < 0:

            self.best_auc = auc

            torch.save(
                model.state_dict(),
                self.save_path
            )

            print(
                f"保存最佳模型 AUC={auc:.4f}"
            )

            return

        # AUC提升
        if auc > self.best_auc + self.min_delta:

            self.best_auc = auc

            self.counter = 0

            torch.save(
                model.state_dict(),
                self.save_path
            )

            print(
                f"AUC提升至 {auc:.4f}，保存模型"
            )

        # AUC未提升
        else:

            self.counter += 1

            print(
                f"AUC未提升 ({self.counter}/{self.patience})"
            )

            if self.counter >= self.patience:

                self.early_stop = True

                print(
                    f"\nEarly Stopping触发！"
                    f"\n最佳AUC={self.best_auc:.4f}"
                )