r"""@package motsfinder.lockfile.lockfile

Class to coordinate concurrent file access.

See the LockFile class for details and examples.
"""

from datetime import datetime, timedelta
import errno
import os
import os.path as op
import socket
import time
import uuid


__all__ = [
    "LockFile",
]


class LockFileError(Exception):
    r"""Base exception for any lockfile related errors."""
    pass

class DanglingLockfile(LockFileError):
    r"""Raised when an unknown lockfile is present which we can't handle."""
    pass

class AlreadyLockedError(LockFileError):
    r"""Raised when we try to lock while already having a lock."""
    pass


class LockFile():
    r"""Locking class used as a context manager.

    This class should be used in a ``with`` statement. It is instantiated with
    a filename of a file you wish to have exclusive access to for a limited
    time (default 10 seconds). Other processes, possibly even on different
    machines accessing the same network file system, will wait until either:
        * the lock is removed
        * or the `lockduration` is exceeded

    Of course, this assumes all processes use this locking mechanism. No
    attempt is made to limit access to the file for processes not using this
    class.

    @b Examples

    ```
        import time
        from motsfinder.lockfile import LockFile

        with LockFile("some_file_name.npy", verbosity=1):
            # Perform some task on the file, knowing that no one else will
            # access the file (provided they also use LockFile)
            time.sleep(5)
    ```

    @b Notes

    The idea is based on a comment in the GNU/Linux open(2) manpage and
    inspired by the [flufl.lock](https://gitlab.com/warsaw/flufl.lock) locking
    module by Barry Warsaw, which has similar goals but many more features.

    The idea is to have each process write a unique file. This should not fail
    even if performed on many nodes concurrently, since every process uses a
    randomized unique name. Acquiring a lock then consists of creating a hard
    link to this file with the name of the actual lock file. This will fail
    for all but one process in concurrent situations even across nodes on an
    NFS share (as is indicated by the manpage comment mentioned above). The
    processes that didn't get the lock then wait a few seconds and try again.

    There is an expiration time stored as access/modified time of the lock
    file. This is `lockduration` seconds in the future. Once this time has
    expired, the lock will be removed as it is assumed the process that
    created that lock hangs, has crashed, lost connection to the NFS, etc.
    """

    NOT_EXIST_ERRORS = (errno.ESTALE, errno.ENOENT)

    def __init__(self, fname, lockduration=10, verbosity=0):
        r"""Create a lock object.

        The actual lock is acquired once the ``with`` context is entered.

        @param fname
            Filename to lock. This file may or may not exist. It will not be
            modified at all and is just used to signal the *intention* to
            coordinate access to this file. The actual lock file will be
            called ``"{fname}.lock"``, where ``{fname}`` is the value provided
            here. This lock file should not exist (except when created by this
            class).
        @param lockduration
            Maximum time in seconds to guarantee the lock. The lock is removed
            before that time if the ``with`` statement is exited before.
            Default is ``10``.
        @param verbosity
            How verbose to be. If `0`, be quiet in all cases (except when
            errors occur, of course). For `1`, only print something if
            acquiring the lock fails and we have to wait. For `2`, print more
            status infos like which files are used, when they are created and
            deleted, etc. Default is `0`.
        """
        self._verbosity = verbosity
        self._lockduration = lockduration
        fname = op.normpath(fname)
        self._basedir = op.dirname(fname)
        self._lockfilename = "%s.lock" % op.basename(fname)
        self._uniquename = ",".join((
            self._lockfilename, socket.getfqdn(), str(os.getpid()),
            str(uuid.uuid4())[-12:],
        ))
        self._uniquefile_created = False
        self._p(" File to lock: %s" % fname)
        self._p("Lockfile name: %s" % self._lockfilename)
        self._p("  Unique name: %s" % self._uniquename)

    @property
    def uniquefile(self):
        r"""The unique file used as second link target."""
        return op.join(self._basedir, self._uniquename)

    @property
    def lockfile(self):
        r"""The file used as primary link target."""
        return op.join(self._basedir, self._lockfilename)

    def __enter__(self):
        r"""Enter the `with` context and establish the lock."""
        try:
            self._lock()
        except:
            self._unlock()
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        r"""Exit the `with` context and remove the lock."""
        self._unlock()
        # Let all exceptions through by not returning True.

    def __del__(self):
        r"""Called upon object destruction.

        There is no guarantee this ever gets called. However, if it does get
        called, it is clear we don't want to keep any lock.
        """
        if hasattr(self, "_uniquefile_created"):
            self._unlock()
        else:
            # When instance attributes don't exist, we probably had an error
            # in the construction process (like an invalid argument to
            # __init__()). In that case, there's no chance we have a unique
            # file or a lock to clean up.
            pass

    def is_locked(self):
        r"""Find out if we currently have the lock.

        The condition to have the lock is that our unique file and the lock
        file are two hard links to the same file. This is checked by making
        sure the hard link count of our unique file is 2 and that their
        contents are equal.

        No check for expiration of the lock is done here.
        """
        if not self._uniquefile_created:
            return False
        links = self._count_links()
        return links == 2 and self._read_lockfile() == self._uniquename

    def _lock(self):
        r"""Acquire the lock.

        This creates the unique file (if it doesn't exist) and then tries to
        create the second hard link. If unsuccessful, we wait a bit and try
        again until the "foreign" lock has expired or has been removed.
        """
        self._create_unique_file()
        while True:
            try:
                self._extend_expiration_time() # in case we had to wait...
                os.link(self.uniquefile, self.lockfile)
                self._p("Lock successfully acquired.")
                return
            except OSError as e:
                if e.errno in self.NOT_EXIST_ERRORS:
                    # Didn't work for some reason. Try again in a bit.
                    pass
                elif e.errno == errno.EEXIST:
                    links = self._count_links()
                    if links == 2 and self._read_lockfile() == self._uniquename:
                        raise AlreadyLockedError("Lock already exists.")
                else:
                    # An unexpected error occurred.
                    self._remove_unique_file()
                    raise
            # Release expired lock of the others.
            if self._has_expired():
                self._break_foreign_lock()
                time.sleep(2)
                continue
            # No luck getting the lock.
            self._p("Locked by someone else. Waiting to retry...", level=1)
            time.sleep(5)

    def _count_links(self):
        r"""Count the number of hard links to our unique file."""
        try:
            return os.stat(self.uniquefile).st_nlink
        except OSError as e:
            if e.errno in self.NOT_EXIST_ERRORS:
                return 0 # no links, i.e. file does not exist
            raise

    def _has_expired(self):
        r"""Check if the currently active lock has expired.

        Returns `False` if there is no lock. This checks the actual lock file,
        i.e. it applies if we have the lock or a different process has it.
        """
        try:
            expires = datetime.fromtimestamp(
                os.stat(self.lockfile).st_mtime
            )
        except OSError as e:
            if e in self.NOT_EXIST_ERRORS:
                return False
            raise
        return datetime.now() > expires

    def _break_foreign_lock(self):
        r"""Break a lock of a different process.

        This should be called when the lock has expired, though no check is
        made here to ensure it actually has expired. It will remove the lock
        file and also the unique file that this lock file pointed to (if
        possible).
        """
        # If the foreign process did not crash and just takes a bit longer
        # than expected, this may pull the rug from under their feet by
        # removing the lock they think they're perfectly fine with. As
        # everywhere in this class, we have to be quite fault tolerant to cope
        # with this.
        self._extend_expiration_time(self.lockfile)
        self._p("Breaking foreign lock.", level=1)
        other_unique = self._read_lockfile()
        if (other_unique
                and not self._is_valid_unique_fname(other_unique)):
            raise DanglingLockfile(
                "Lock file does not point to unique file."
            )
        self._unlink(self.lockfile)
        if other_unique:
            self._unlink(op.join(self._basedir, other_unique))

    def _is_valid_unique_fname(self, fname):
        r"""Validate that the given filename is a valid "unique" file name.

        We don't ever want to delete a file not managed by this class. We
        therefore check that the unique file name adheres to our naming
        convention by starting with the lock file name (without path) but
        being longer than that. This should remove any chance of accidental
        file deletion (unless someone actively tries to make us look bad).
        """
        return (fname.startswith(self._lockfilename)
                and len(fname) > len(self._lockfilename))

    def _read_lockfile(self):
        r"""Return the contents of the current lock file.

        If there is no lock file, returns `None`. If we created the lock file,
        the contents equals the name (without path) of our unique file.
        """
        try:
            with open(self.lockfile) as f:
                return f.read()
        except EnvironmentError as e:
            if e.errno in self.NOT_EXIST_ERRORS:
                return None
            raise

    def _extend_expiration_time(self, fname=None):
        r"""Set the expiration time of the unique file to the future.

        This also affects the lock file in case we currently have the lock.
        The optional parameter can be used to extend the expiration time of
        e.g. a foreign lock file.
        """
        if fname is None:
            fname = self.uniquefile
        future = datetime.now() + timedelta(seconds=self._lockduration)
        expires = time.mktime(future.timetuple())
        try:
            os.utime(fname, (expires, expires))
        except OSError as e:
            if e.errno not in self.NOT_EXIST_ERRORS:
                raise

    def _create_unique_file(self):
        r"""Write the unique file and set its expiration time."""
        with open(self.uniquefile, 'w') as f:
            f.write(self._uniquename)
        self._uniquefile_created = True
        self._extend_expiration_time()
        self._p("Unique file created: %s" % self.uniquefile)

    def _remove_unique_file(self):
        r"""Remove our unique file.

        This does nothing if there is no unique file.
        """
        if self._uniquefile_created:
            self._unlink(self.uniquefile)
            self._uniquefile_created = False
            self._p("Unique file deleted: %s" % self.uniquefile)

    def _unlock(self):
        r"""Release the lock (if any).

        This will *not* fail if we currently don't have the lock, which may
        easily happen when a different process removes our lock after it has
        expired.

        The unique file is removed in all cases.
        """
        if self.is_locked():
            self._unlink(self.lockfile)
            self._remove_unique_file()
            self._p("Lock removed.")
        else:
            self._remove_unique_file()

    def _p(self, *args, level=2, **kwargs):
        r"""Print a message depending on its level and our verbosity."""
        if self._verbosity >= level:
            print(*args, **kwargs)

    @classmethod
    def _unlink(cls, fname):
        r"""Fault-tolerant file deletion.

        Does nothing if the file does not exist.
        """
        try:
            os.unlink(fname)
        except OSError as e:
            if e.errno not in cls.NOT_EXIST_ERRORS:
                raise
