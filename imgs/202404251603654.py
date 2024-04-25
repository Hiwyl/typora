#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author :          lance
@Email : lance.wang@vastaitech.com
@Time  : 	2024/03/21 20:13:20
'''
import contextlib
import os
from multiprocessing import Event, Process, Queue
from typing import Literal, Optional, Tuple

import psutil
from loguru import logger
from vamc.utils import Timer
from vamc.vacc import BuildLLMModel

MemUnitType = Literal['GiB', 'MiB', 'KiB']
# mp.set_start_method('spawn')


def bytes_to_target_unit(mem_bytes: int, unit: MemUnitType) -> float:
    units = {'GiB': 1 << 30, 'MiB': 1 << 20, 'KiB': 1 << 10}
    _rename_map = {'GB': 'GiB', 'MB': 'MiB', 'KB': 'KiB'}
    if unit not in units:
        unit = _rename_map[unit]
    return float(mem_bytes) / units[unit]


def host_memory_info(pid: Optional[int] = None) -> Tuple[int, int]:
    # FIXME:(lance.0424) why will psutil.AccessDenied?
    # try:
    #     process = psutil.Process(pid)
    # except Exception as e:
    #     logger.error(e)
    process = psutil.Process()
    # USS reports the amount of memory that would be freed if the process
    # was terminated right now.
    #   https://psutil.readthedocs.io/en/latest/index.html#psutil.Process.memory_full_info
    vmem = psutil.virtual_memory()
    total_mem = vmem.total
    alloc_mem = process.memory_full_info().uss
    return alloc_mem, total_mem  # used, total


def device_memory_info() -> Tuple[int, int]:
    return 0, 0


class MemoryMonitor:

    def __init__(self, query_interval=0.1, disable_host_mem_monitor=False, start=False):
        self.query_interval = query_interval  # second(s)
        self.mem_monitor_process = None
        # bytes
        self._peak_host_memory = 0
        self._peak_device_memory = 0

        self.pid = os.getpid()
        self.device_handles = {}

        self.signal_event = Event()  # Sending signal to subprocess
        self.peak_mem_queue = Queue()  # Receiving results from subprocess

        self.disable_host_mem_monitor = disable_host_mem_monitor

        if start:
            self.start()

    def start(self):
        self.mem_monitor_process = Process(
            target=self._upd_peak_memory_usage, args=(self.signal_event, self.peak_mem_queue)
        )
        self.mem_monitor_process.start()
        logger.debug("Launched memory monitor subprocess.")

    def kill(self):
        if self.mem_monitor_process is not None:
            self.mem_monitor_process.kill()
            logger.debug("Memory monitor subprocess is killed.")

    def stop(self):
        self.signal_event.set()
        logger.debug("Sent signal to stop memory monitor subprocess.")
        peak_mem_use = self.peak_mem_queue.get(timeout=20)

        self._peak_host_memory = max(self._peak_host_memory, peak_mem_use[0])
        self._peak_device_memory = max(self._peak_device_memory, peak_mem_use[1])

        self.mem_monitor_process.join(timeout=20)
        self.mem_monitor_process = None
        logger.debug("Memory monitor subprocess joined.")

    def _upd_peak_memory_usage(self, signal_event, peak_mem_queue):
        peak_host_used, peak_device_used = self.get_memory_usage()
        while not signal_event.is_set():
            host_used, device_used = self.get_memory_usage()
            peak_host_used = max(host_used, peak_host_used)
            peak_device_used = max(device_used, peak_device_used)
        peak_mem_queue.put((peak_host_used, peak_device_used))

    def get_memory_usage(self):
        if self.disable_host_mem_monitor:
            host_used = 0
        else:
            host_used, _ = host_memory_info(self.pid)
        device_used, _ = device_memory_info()
        return host_used, device_used

    def get_peak_memory_usage(self, unit: MemUnitType = 'GiB'):
        return bytes_to_target_unit(self._peak_host_memory, unit), bytes_to_target_unit(
            self._peak_device_memory, unit
        )


@contextlib.contextmanager
def monitor_handler():
    # monitor start
    tmr = Timer()
    memory_monitor = MemoryMonitor()
    memory_monitor.start()
    try:
        yield
    except Exception as e:
        raise Exception(e)
    finally:
        # monitor stop
        memory_monitor.stop()
        peak_cpu_used, peak_gpu_used = memory_monitor.get_peak_memory_usage("GiB")
        cost_time = tmr.since_start() / 60  # mins

        logger.info(
            f"Time Cost : {cost_time:.3f} mins, Host Mem Max : {peak_cpu_used :.3f} GiB, Device Mem Max : {peak_gpu_used :.3f} GiB."
        )


@monitor_handler()
def compile(cfg_path: str):
    BuildLLMModel(cfg_path).build()


if __name__ == "__main__":
    compile("./cfg.yaml")
