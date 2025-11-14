#!/usr/bin/env python3
import argparse
import time
from datetime import datetime
import os
import psutil
import signal
import sys


def format_datetime(dt):
    return dt.strftime('%Y-%m-%d %H:%M:%S.%f')

def format_timestamp(timestamp : float):
    return format_datetime(datetime.fromtimestamp(timestamp))


def log(msg):
    print(f"[{format_timestamp(time.time())}] {msg}")
    sys.stdout.flush()


def get_memused_bytes():
    vm = psutil.virtual_memory()
    return vm.total - vm.available


def pretty_bytes(bytes: int) -> str:
    return f'{bytes/1024**3:.2f} GB ({bytes} bytes)'


def run(args: argparse.Namespace):
    initial_mem = None
    max_mem = 0
    now = time.time()
    start_time = now    
    max_time = None
    running = True

    # Auxiliary functions

    def handle_exit(signum=None, frame=None):
        nonlocal running
        """Print max MemUsed before exiting."""
        if not running:
            return
        running = False
        print("\n" + "=" * 60)
        if initial_mem is not None:
            print(f'Initial MemUsed: {pretty_bytes(initial_mem)} at {format_timestamp(start_time)}')
        if max_time is not None:
            print(
                f'Max MemUsed: {pretty_bytes(max_mem)} at {format_timestamp(max_time)}')
        else:
            print("No memory data collected.")
        print("=" * 60)
        sys.stdout.flush()
        sys.exit(0)
    
    def write_summary():
        if args.summary is None:
            return
        if max_time is not None:
            mem_used_str = f'Max MemUsed: {pretty_bytes(max_mem)} at {format_timestamp(max_time)}'
        else:
            mem_used_str = "No memory data collected."

        # Write to tmp and atomically replace to prevent invalid output if program is killed during writing.
        summary_tmp = args.summary + '.tmp' 
        with open(summary_tmp, 'w') as f:
            f.write(
                f'''Started: {format_timestamp(start_time)}
MemTotal: {pretty_bytes(mem_total_bytes)}
Initial MemUsed: {pretty_bytes(initial_mem)}
Last update: {format_timestamp(now)}
{mem_used_str}''')
        os.replace(summary_tmp, args.summary)

    # Register signal handlers for clean shutdown (e.g., SLURM job end)
    # TODO: doesn't work for scancel!
    signal.signal(signal.SIGTERM, handle_exit)
    signal.signal(signal.SIGINT, handle_exit)

    mem_total_bytes = psutil.virtual_memory().total
    initial_mem = get_memused_bytes()
    log(f'MemTotal: {pretty_bytes(mem_total_bytes)}')

    t_last_print = 0
    try:
        while running:
            now = time.time()

            mem_used = get_memused_bytes()

            if mem_used > max_mem:
                max_mem = mem_used
                max_time = now

            if args.print_interval and now - t_last_print >= args.print_interval:
                log(f'MemUsed: {pretty_bytes(mem_used)}')
                write_summary()
                t_last_print = now

            time.sleep(args.update_interval)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        handle_exit()


def main():
    global start_time, max_mem, max_time, running

    p = argparse.ArgumentParser(
        description="Monitor MemUsed = (MemTotal - MemAvailable) using psutil.")
    p.add_argument("--update-interval", type=float, default=0.1,
                   help="Interval (s) to check memory usage.")
    p.add_argument("--print-interval", type=float, default=1.0,
                   help="Interval (s) to print memory usage.")
    p.add_argument("--summary", type=str, help="Path to summary output file.")
    args = p.parse_args()

    log(f"Started MemUsed watcher (pid={os.getpid()}, args={args})")
    run(args)


if __name__ == "__main__":
    main()
