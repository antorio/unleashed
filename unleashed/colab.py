"""Disconnecting the Colab runtime after a render (Face Swap: "Disconnect
Colab when done").

The app cannot do it itself: google.colab.runtime.unassign() works only in
the notebook's kernel, and the app runs as a separate process (run.py). The
notebook's "Jalankan Unleashed" cell sets UNLEASHED_DISCONNECT_FLAG to a file
path and checks that file every second. The app writes the moment to
disconnect into it; at that moment the cell stops the app, flushes Google
Drive (drive.flush_and_unmount(), so the saved videos reach Drive) and
unassigns the runtime. Removing the file cancels.
"""
import os
import time

DRIVE_ROOT = '/content/drive'
DELAY = 60                  # seconds to change one's mind (Stay connected)


def flag_path():
    return os.environ.get('UNLEASHED_DISCONNECT_FLAG') or None


def available():
    """Started by a notebook cell that can disconnect."""
    return bool(flag_path())


def on_drive(folder, drive_root=DRIVE_ROOT):
    real = os.path.realpath(folder or '')
    root = os.path.realpath(drive_root)
    return real.startswith(root + os.sep)


def check(entries, finished, stopped, out_dir, drive_root=DRIVE_ROOT):
    """(True, '') when the run may end the session, else (False, why). Every
    file of the run must be there, not empty, in a folder on Google Drive (the
    runtime's own disk goes away with it)."""
    if stopped:
        return False, 'the render was stopped'
    if len(finished) < len(entries):
        return False, 'not every file was saved'
    for path in finished:
        try:
            if os.path.getsize(path) <= 0:
                return False, f'{os.path.basename(path)} is empty'
        except OSError:
            return False, f'{os.path.basename(path)} is not there'
    if not on_drive(out_dir, drive_root):
        return False, 'the output folder is not on Google Drive, so the files would go with the runtime'
    return True, ''


def schedule(delay=DELAY):
    path = flag_path()
    if not path:
        return False
    tmp = path + '.part'
    with open(tmp, 'w') as fh:
        fh.write(f'{time.time() + delay:.0f}')
    os.replace(tmp, path)
    return True


def cancel():
    path = flag_path()
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def pending():
    path = flag_path()
    return bool(path) and os.path.exists(path)
