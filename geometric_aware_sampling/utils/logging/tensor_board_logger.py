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
            buffer_time: The interval (in seconds) to flush the buffer and forward the data.
        """
        self.forward_thread = None
        self.class_context = class_context
        self.buffer_time = buffer_time
        self.buffer = StringIO()
        self.lock = threading.Lock()
        self.running = False

        self.wall_time = time.time()
        self.method_name = method_name
        self.step_counter = 0

    def write(self, message):
        with self.lock:
            self.buffer.write(message)
            self.buffer.flush()

    def isatty(self):
        return False  # Mimics the behavior of a non-terminal file-like object.

    def flush(self):
        pass  # Required for `sys.stdout` compatibility.

    def start_forwarding(self):
        """Start the background thread to forward logs periodically."""
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
        with self.lock:
            output = self.buffer.getvalue()

            # print to self._stdout
            print(output, end="", flush=True, file=self._stdout)

            if output.strip():  # Avoid empty logs.
                self.class_context.tensorboard_logger.writer.add_text(
                    f"{self.wall_time}_{self.method_name}",
                    output,
                    global_step=self.step_counter,
                )
                self.step_counter += 1
                self.buffer = StringIO()  # Reset buffer.

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self
        self.start_forwarding()
        return self

    def __exit__(self, *args):
        self.stop_forwarding()
        sys.stdout = self._stdout


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
