# Import the required modules
import os
import time
import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


# Create a new logging class that inherits from FileSystemEventHandler
class LoggingEventHandler(FileSystemEventHandler):
    def __init__(self):
        self.logged_events = set()

    # Check if a create new file/directory condition has been met and not log if it already exists
    def on_created(self, event):
        path = event.src_path
        if path in self.logged_events:
            return
        self.logged_events.add(path)
        if event.is_directory:
            log_action("Directory Created", path)
        else:
            log_action("File Created", path)

    # Check if a delete file/directory condition has been met and not log if it already exists
    def on_deleted(self, event):
        path = event.src_path
        if path in self.logged_events:
            return
        self.logged_events.add(path)
        if event.is_directory:
            log_action("Directory Deleted", path)
        else:
            log_action("File Deleted", path)


# Logging timing
def log_action(action, path):
    current_time = str(datetime.datetime.now())
    message = f'{current_time} - {action} - {path}'
    with open('pc_actions.log', 'a') as f:
        f.write(message + '\n')


# Running the program

if __name__ == '__main__':
    path = '.'
    event_handler = LoggingEventHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
