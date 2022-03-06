from typing import Dict, List, Sequence, Tuple, Union
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.data import ConcatDataset, Dataset

from spot_vrl.data import ImuData


class SingleTerrainDataset(Dataset[torch.Tensor]):
    """IMU dataset from a single terrain type.

    Datum are organized as fixed-sized matrices where rows encode individual
    data types over time and columns encode observations of all data types.
    """

    def __init__(
        self, path: Union[str, Path], start: float = 0, end: float = np.inf
    ) -> None:
        imu = ImuData(path)

        window_size = 40  # approx 3 seconds at 13.6 Hz
        self.windows: List[torch.Tensor] = []

        ts, data = imu.query_time_range(imu.all_sensor_data, start, end)
        # Overlap windows for more data points
        for i in range(0, data.shape[-1] - window_size + 1, window_size // 8):
            window = data[:, i : i + window_size]

            # Add statistical features as additional column vectors
            mean = window.mean(axis=1)[:, np.newaxis]
            std = window.std(axis=1)[:, np.newaxis]
            med = np.median(window, axis=1)[:, np.newaxis]
            # TODO(eyang): Joint data is periodic, so these features may not be
            # very useful. May want to use quantiles, IQR, min, max, etc.

            window = np.hstack((window, mean, std, med))
            self.windows.append(torch.from_numpy(window))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.windows[index]


Triplet = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class ManualTripletDataset(Dataset[Triplet]):
    """Hardcoded triplet training dataset using fixed log files and time
    ranges."""

    def __init__(self) -> None:
        concretes = [
            SingleTerrainDataset(
                "data/2022-02-27/2022-02-27-16-31-16.bddf",
                start=1646001080,
                end=1646001382,
            ),
            SingleTerrainDataset(
                "data/2022-02-27/2022-02-27-17-55-35.bddf",
                start=1646006138,
                end=1646006272,
            ),
        ]

        grasses = [
            SingleTerrainDataset(
                "data/2022-02-27/2022-02-27-16-47-20.bddf",
                start=1646002045,
                end=1646002323,
            ),
            SingleTerrainDataset(
                "data/2022-02-27/2022-02-27-17-50-10.bddf",
                start=1646005815,
                end=1646006071,
            ),
        ]

        sml_rocks = [
            SingleTerrainDataset(
                "data/2022-02-27/2022-02-27-17-24-55.bddf",
                start=1646004299,
                end=1646004528,
            ),
            SingleTerrainDataset(
                "data/2022-02-27/2022-02-27-17-32-28.bddf",
                start=1646004753,
                end=1646005015,
            ),
        ]

        self._categories: Dict[str, ConcatDataset[torch.Tensor]] = {}

        self._categories["concrete"] = ConcatDataset(concretes)
        self._categories["grass"] = ConcatDataset(grasses)
        self._categories["sml_rock"] = ConcatDataset(sml_rocks)

        for cat, ds in self._categories.items():
            logger.info(f"{cat} data points: {len(ds)}")

        self._cumulative_sizes: List[int] = np.cumsum(
            [len(ds) for ds in self._categories.values()], dtype=np.int_
        ).tolist()

        self._rng = np.random.default_rng()

    def _get_random_datum(self, cats: Sequence[str] = ()) -> torch.Tensor:
        """Returns a random datum from the specified categories.

        The category is chosen from the sequence with equal probability.

        Args:
            cats (tuple[str] | list[str]): A sequence containing the categories
                to sample from. If the sequence is empty, all of the internal
                categories are used instead.
        """
        if not cats:
            cats = tuple(self._categories.keys())

        cat: str = self._rng.choice(cats)
        ds = self._categories[cat]
        datum: torch.Tensor = ds[self._rng.integers(len(ds))]
        return datum

    def __len__(self) -> int:
        return self._cumulative_sizes[-1] * 3

    def __getitem__(self, index: int) -> Triplet:
        index = index % self._cumulative_sizes[-1]

        cat_idx: int = np.searchsorted(
            self._cumulative_sizes, index, side="right"
        ).astype(int)

        if cat_idx > 0:
            index -= self._cumulative_sizes[cat_idx - 1]

        cat_names = tuple(self._categories.keys())

        anchor = self._categories[cat_names[cat_idx]][index]
        pos = self._get_random_datum((cat_names[cat_idx],))
        neg = self._get_random_datum((*cat_names[:cat_idx], *cat_names[cat_idx + 1 :]))

        return anchor, pos, neg


class ManualTripletHoldoutSet:
    def __init__(self) -> None:
        concretes = [
            SingleTerrainDataset(
                "data/2022-02-08/2022-02-08-17-52-06.bddf",
                start=1644364331,
                end=1644364350,
            ),
            SingleTerrainDataset(
                "data/2022-02-08/2022-02-08-17-52-06.bddf",
                start=1644364378,
                end=1644364384,
            ),
            SingleTerrainDataset(
                "data/2022-02-08/2022-02-08-17-48-11.bddf",
                start=1644364103,
                end=1644364131,
            ),
            SingleTerrainDataset(
                "data/2022-02-27/2022-02-27-16-38-12.bddf",
                start=1646001496,
                end=1646001572,
            ),
        ]

        grasses = [
            SingleTerrainDataset(
                "data/2022-02-08/2022-02-08-17-52-06.bddf",
                start=1644364352,
                end=1644364374,
            ),
            SingleTerrainDataset(
                "data/2022-02-08/2022-02-08-17-48-11.bddf",
                start=1644364133,
                end=1644364148,
            ),
            SingleTerrainDataset(
                "data/2022-02-27/2022-02-27-16-43-41.bddf",
                start=1646001826,
                end=1646001946,
            ),
        ]

        sml_rocks = [
            SingleTerrainDataset(
                "data/2022-02-27/2022-02-27-17-40-02.bddf",
                start=1646005206,
                end=1646005254,
            ),
            SingleTerrainDataset(
                "data/2022-02-27/2022-02-27-17-41-31.bddf",
                start=1646005295,
                end=1646005347,
            ),
        ]

        self._categories: Dict[str, ConcatDataset[torch.Tensor]] = {}

        self._categories["concrete"] = ConcatDataset(concretes)
        self._categories["grass"] = ConcatDataset(grasses)
        self._categories["sml_rock"] = ConcatDataset(sml_rocks)

        for cat, ds in self._categories.items():
            logger.info(f"{cat} data points: {len(ds)}")
