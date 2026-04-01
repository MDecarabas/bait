"""
Configuration file watcher for hot-reloading.
"""

import time
from pathlib import Path
from threading import Thread, Event
from typing import Callable, Optional

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer


class ConfigFileHandler(FileSystemEventHandler):
    """Handler for config file changes."""

    def __init__(self, config_path: Path, callback: Callable, debounce_seconds: float = 0.5):
        self.config_path = config_path.resolve()
        self.callback = callback
        self.debounce_seconds = debounce_seconds
        self.last_modified = 0.0

    def on_modified(self, event: FileSystemEvent):
        """Called when a file is modified."""
        if event.is_directory:
            return

        # Check if it's our config file
        event_path = Path(event.src_path).resolve()
        if event_path == self.config_path:
            self._trigger_reload()

    def on_created(self, event: FileSystemEvent):
        """Called when a file is created."""
        if event.is_directory:
            return

        # Check if it's our config file
        event_path = Path(event.src_path).resolve()
        if event_path == self.config_path:
            self._trigger_reload()

    def _trigger_reload(self):
        """Trigger reload with debouncing."""
        current_time = time.time()

        # Debounce: only reload if enough time has passed
        if current_time - self.last_modified < self.debounce_seconds:
            return

        self.last_modified = current_time

        print(f"🔄 Config file changed: {self.config_path.name}")
        print("   Reloading configuration...")

        try:
            self.callback()
            print("✅ Configuration reloaded successfully!")
        except Exception as e:
            print(f"❌ Error reloading configuration: {e}")


class ConfigWatcher:
    """Watches config file for changes and triggers reload."""

    def __init__(self, config_path: str = "config.yaml", callback: Optional[Callable] = None):
        self.config_path = Path(config_path).resolve()
        self.callback = callback
        self.observer: Optional[Observer] = None
        self.running = Event()

    def start(self):
        """Start watching the config file."""
        if self.observer is not None and self.observer.is_alive():
            print("⚠️  Config watcher is already running")
            return

        # Watch the directory containing the config file
        watch_dir = self.config_path.parent

        event_handler = ConfigFileHandler(self.config_path, self.callback or self._default_callback)

        self.observer = Observer()
        self.observer.schedule(event_handler, str(watch_dir), recursive=False)
        self.observer.start()
        self.running.set()

        print(f"👁️  Watching config file: {self.config_path}")

    def stop(self):
        """Stop watching the config file."""
        if self.observer is None:
            return

        self.running.clear()
        self.observer.stop()
        self.observer.join()
        self.observer = None

        print("🛑 Config watcher stopped")

    def _default_callback(self):
        """Default callback when no callback is provided."""
        print("   No callback registered for config reload")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


# Global watcher instance
_watcher: Optional[ConfigWatcher] = None


def start_config_watcher(config_path: str = "config.yaml", callback: Optional[Callable] = None):
    """Start the global config watcher."""
    global _watcher

    if _watcher is not None:
        print("⚠️  Config watcher already started")
        return _watcher

    _watcher = ConfigWatcher(config_path, callback)
    _watcher.start()
    return _watcher


def stop_config_watcher():
    """Stop the global config watcher."""
    global _watcher

    if _watcher is None:
        return

    _watcher.stop()
    _watcher = None


def get_watcher() -> Optional[ConfigWatcher]:
    """Get the global watcher instance."""
    return _watcher
