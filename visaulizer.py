from pytorch_lightning.callbacks import Callback
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
import torch

class RichProgressBar(Callback):
    def __init__(self):
        self.console = Console()
        self.train_progress = None
        self.val_progress = None
        self.loss_table = Table(show_header=True, header_style="bold magenta")
        self.loss_table.add_column("Epoch", justify="right")
        self.loss_table.add_column("g_loss", justify="right")
        self.loss_table.add_column("L1_loss", justify="right")
        self.loss_table.add_column("SSIM_loss", justify="right")
        self.loss_table.add_column("vgg_loss", justify="right")
        self.loss_table.add_column("hwt_loss", justify="right")
        self.loss_table.add_column("total_loss", justify="right")
        self.loss_table.add_column("d_loss", justify="right")
        self.loss_table.add_column("val_loss", justify="right")
        self.loss_table.add_column("val_l1", justify="right")

    def on_train_start(self, trainer, pl_module):
        columns = [
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ]
        self.train_progress = Progress(*columns)
        self.train_task = self.train_progress.add_task("[cyan]Training", total=trainer.num_training_batches)
        self.train_progress.start()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.train_progress.update(self.train_task, advance=1)

        # 更新损失表格
        epoch = trainer.current_epoch + 1
        losses = outputs['losses']
        val_loss = trainer.callback_metrics.get('val_loss', float('nan'))
        val_l1 = trainer.callback_metrics.get('val_l1', float('nan'))

        self.loss_table.add_row(
            str(epoch),
            f"{losses.get('g_loss', float('nan')):.3f}",
            f"{losses.get('L1_loss', float('nan')):.3f}",
            f"{losses.get('SSIM_loss', float('nan')):.3f}",
            f"{losses.get('vgg_loss', float('nan')):.3f}",
            f"{losses.get('hwt_loss', float('nan')):.3f}",
            f"{losses.get('total_loss', float('nan')):.3f}",
            f"{losses.get('d_loss', float('nan')):.3f}",
            f"{val_loss:.3f}" if not torch.isnan(val_loss) else "N/A",
            f"{val_l1:.3f}" if not torch.isnan(val_l1) else "N/A"
        )

        self.console.clear_live()
        self.console.begin_live(self.loss_table, auto_refresh=False)
        self.console.print(self.train_progress)

    def on_validation_end(self, trainer, pl_module):
        pass

    def on_train_end(self, trainer, pl_module):
        self.train_progress.stop()
        self.console.end_live()


        