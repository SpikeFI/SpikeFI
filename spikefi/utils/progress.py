# This file is part of SpikeFI.
# Copyright (C) 2024 Theofilos Spyrou, Sorbonne Université, CNRS, LIP6

# SpikeFI is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# SpikeFI is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.


import math
from itertools import cycle
import re
from threading import Lock
from time import sleep, time
from typing import Any
from tqdm import tqdm


class CampaignProgress:
    def __init__(
            self,
            batches_num: int,
            rounds_num: int,
            epochs_num: int | None = None
    ) -> None:
        self.is_training = bool(epochs_num)
        self.loss = 0.
        self.accu = 0.
        self.status = 0.
        self.batch = 0
        self.batch_num = batches_num
        self.round = 0
        self.round_num = rounds_num
        self.epoch = 0
        self.epoch_num = epochs_num
        self.iter = 0
        self.iter_num = batches_num * rounds_num * (epochs_num or 1)
        self.fragment = 1. / self.iter_num
        self.start_time = 0.
        self.end_time = 0.

        self.pbar = tqdm(total=self.iter_num, leave=False)

        self._flush_lines_num = 0
        self._loading_bar = cycle(['-', '-', '\\', '\\', '|', '|', '/', '/'])

        self._lock = Lock()

    def __getstate__(self) -> Any:
        state = self.__dict__.copy()
        del state['_lock']
        return state

    def __setstate__(self, state: Any) -> None:
        self.__dict__.update(state)
        self._lock = Lock()

    def __str__(self) -> str:
        s_status = (
            "Status: "
            + (
                ("running " + next(self._loading_bar))
                if not self.has_finished() else "done"
            ) + "\n"
        )

        s_header = (
            "|   Batch #   "
            "|   Round #   "
            "|  Total time  "
            "|  Progress  |\n"
        )

        with self._lock:
            if self.is_training:
                s_header = (
                    "|    Loss    "
                    "|  Accuracy  "
                    "|   Epoch #   "
                ) + s_header
                s_border = re.sub(r'[^+\n]', '-', s_header.replace('|', '+'))

                s = s_status + s_border + s_header
                s += f"|  {self.loss:7.3f}   |"
                s += f"  {self.accu * 100.:6.3f} %  |"
                s += f"  {self.epoch:4d}/{self.epoch_num:<4d}  "
            else:
                s_border = re.sub(r'[^+\n]', '-', s_header.replace('|', '+'))
                s = s_status + s_border + s_header

            s += f"| {self.batch:5d}/{self.batch_num:<5d} |"
            s += f" {self.round:5d}/{self.round_num:<5d} |"
            if self.start_time:
                s += f" {(time() - self.start_time):8.1f} sec |"
            s += f"  {self.status * 100.:6.2f} %  |"
            s += "\n" + s_border

            self._flush_lines_num = s.count('\n') + 2

            return s

    def has_finished(self) -> bool:
        with self._lock:
            return (
                self.end_time != 0.
                or math.isclose(self.status, 1, abs_tol=1e-12)
            )

    def get_duration_sec(self) -> float | None:
        if self.has_finished():
            with self._lock:
                return self.end_time - self.start_time

        return None

    def show(self, mode: str | None = None) -> None:
        if not mode:
            try:
                get_ipython()  # type: ignore
                mode = 'pbar'
            except NameError:
                mode = 'verbose'

        if mode == 'verbose':
            self._show_table()
            with self._lock:
                self.pbar.refresh()
        elif mode == 'table':
            self._show_table()
        elif mode == 'pbar':
            with self._lock:
                self.pbar.refresh()
        elif mode == 'silent':
            return

    def _show_table(self):
        print('\033[1A\x1b[2K' * self._flush_lines_num)  # Line up, line clear
        print(self)

    def set_train(self, loss: float, accu: float) -> None:
        with self._lock:
            if loss:
                self.loss = loss
            if accu:
                self.accu = accu

    def step(self) -> None:
        with self._lock:
            self.status += self.fragment
            self.iter += 1
            self.pbar.n += 1

    def step_batch(self) -> None:
        with self._lock:
            self.batch = (self.batch % self.batch_num) + 1

    def step_epoch(self) -> None:
        with self._lock:
            self.epoch = (self.epoch % self.epoch_num) + 1

    def step_round(self) -> None:
        with self._lock:
            self.round = (self.round % self.round_num) + 1

    def timer(self) -> None:
        with self._lock:
            if self.start_time and not self.end_time:
                self.end_time = time()
            else:
                self.start_time = time()
                self.end_time = 0.


def refresh_progress_job(
        progress: CampaignProgress,
        interval: float = 0.1,
        mode: str | None = None
) -> None:
    while not progress.has_finished():
        progress.show(mode)
        sleep(interval)

    progress.show(mode)
    with progress._lock:
        progress.pbar.close()
