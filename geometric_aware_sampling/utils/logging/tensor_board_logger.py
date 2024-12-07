import re
import sys
import threading
import time
from abc import ABC
from functools import wraps
from io import StringIO
from types import FunctionType


class TensorBoardLogger:
    def __init__(self, class_context, method_name, buffer_time=1.0):
        """
        Args:
            class_context: The class object that contains the tensorboard_logger attribute.
            method_name: A string identifier for the TensorBoard log entry.
            buffer_time: The interval (in seconds) to flush the buffer and forward the data.
        """
        self.class_context = class_context
        self.method_name = method_name
        self.buffer_time = buffer_time

        self.stdout_buffer = StringIO()
        self.stderr_buffer = StringIO()
        self.stdout_lock = threading.Lock()
        self.stderr_lock = threading.Lock()

        self.running = False
        self.wall_time = time.time()
        self.step_counter = 0

        self.forward_thread = None
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr

        self.tqdm_regex = re.compile(r"\r.*")  # Detect tqdm-style carriage returns

    def write_stdout(self, message):
        """Handle stdout messages."""
        if self.tqdm_regex.match(message):
            # Pass tqdm-style updates directly to the original stdout
            self._original_stdout.write(message)
            self._original_stdout.flush()
        else:
            self._original_stdout.write(message)
            with self.stdout_lock:
                self.stdout_buffer.write(message)

    def write_stderr(self, message):
        """Handle stderr messages."""
        if self.tqdm_regex.match(message):
            # Pass tqdm-style updates directly to the original stderr
            self._original_stderr.write(message)
            self._original_stderr.flush()
        else:
            self._original_stderr.write(message)
            with self.stderr_lock:
                self.stderr_buffer.write(message)

    def flush_stdout(self):
        self._original_stdout.flush()

    def flush_stderr(self):
        self._original_stderr.flush()

    def start_forwarding(self):
        """Start the background thread to periodically flush logs."""
        self.running = True
        self.forward_thread = threading.Thread(
            target=self._forward_periodically, daemon=True
        )
        self.forward_thread.start()

    def stop_forwarding(self):
        """Stop the background thread and flush remaining logs."""
        self.running = False
        if self.forward_thread.is_alive():
            self.forward_thread.join()
        self._flush_to_tensorboard()

    def _forward_periodically(self):
        while self.running:
            time.sleep(self.buffer_time)
            self._flush_to_tensorboard()

    def _flush_to_tensorboard(self):
        """Flush logs from buffers to TensorBoard."""
        with self.stdout_lock:
            stdout_output = self.stdout_buffer.getvalue()
            if stdout_output.strip():
                self.class_context.tensorboard_logger.writer.add_text(
                    f"{self.wall_time}_{self.method_name}_stdout",
                    stdout_output,
                    global_step=self.step_counter,
                )
                self.step_counter += 1
                self.stdout_buffer = StringIO()

        with self.stderr_lock:
            stderr_output = self.stderr_buffer.getvalue()
            if stderr_output.strip():
                self.class_context.tensorboard_logger.writer.add_text(
                    f"{self.wall_time}_{self.method_name}_stderr",
                    stderr_output,
                    global_step=self.step_counter,
                )
                self.step_counter += 1
                self.stderr_buffer = StringIO()

    def __enter__(self):
        sys.stdout = self  # Redirect stdout
        sys.stderr = self  # Redirect stderr
        self.start_forwarding()
        return self

    def __exit__(self, *args):
        self.stop_forwarding()
        sys.stdout = self._original_stdout  # Restore stdout
        sys.stderr = self._original_stderr  # Restore stderr

    def write(self, message):
        """Split writes between stdout and stderr."""
        if sys.stdout == self:
            self.write_stdout(message)
        elif sys.stderr == self:
            self.write_stderr(message)

    def flush(self):
        """Flush both stdout and stderr."""
        self.flush_stdout()
        self.flush_stderr()

    def isatty(self):
        """Return True to indicate that this stream is a terminal."""
        return self._original_stdout.isatty()


def enableLogging(method):
    @wraps(method)
    def wrapped(*args, **kwargs):
        with TensorBoardLogger(
            args[0],
            method.__name__,
        ):
            return method(*args, **kwargs)

    return wrapped


class ClassLogger(type):
    """
    Metaclass that wraps all methods of a class with a logging function
    Based on https://stackoverflow.com/a/11350487
    """

    def __new__(meta, classname, bases, class_dict):

        new_class_dict = {}

        for attributeName, attribute in class_dict.items():

            # skip __init__ and __new__ methods
            if attributeName in ["__init__", "__new__"]:
                new_class_dict[attributeName] = attribute
                continue

            if isinstance(attribute, FunctionType):
                # replace it with a wrapped version
                attribute = enableLogging(attribute)
            new_class_dict[attributeName] = attribute

        return type.__new__(meta, classname, bases, new_class_dict)


class LogEnabledABC(ClassLogger, ABC):
    """
    Abstract Base Class with logging enabled for all methods

    Based on https://stackoverflow.com/a/57351066
    """

    pass
