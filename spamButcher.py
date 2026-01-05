#!/usr/bin/env python
"""Command-line entry point for running the SpamButcher worker without the Flask UI."""
import time

from spam_processing import (
    ConfigManager,
    EmailDatabase,
    EmailProcessor,
    ProcessingMonitor,
    ProcessorWorker,
)


def main() -> None:
    config_manager = ConfigManager()
    monitor = ProcessingMonitor()
    database = EmailDatabase()
    processor = EmailProcessor(config_manager, monitor, database)
    worker = ProcessorWorker(processor, monitor, config_manager)

    worker.start()
    print("SpamButcher worker started. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping worker...")
        worker.stop()


if __name__ == '__main__':
    main()
