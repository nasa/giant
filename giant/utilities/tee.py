


"""
This module provides 2 classes for performing tees of outputs to a file.

A tee (in unix terminology) refers to the ability to pipe 1 input to multiple outputs.  Here that means we can pipe a
single call to stderr or stdout to write to both the standard stderr/stdout, as well as a user specified file.  This is
useful for logging an interactive program (where you want to print both to the screen and to a file).

The :class:`Tee` class is used to tee either STDOUT or STDERR to a file, while the :class:`DoubleTee` class is used to
tee both STDOUT and STDERR to the same file.  Both of these classes can be used as context managers.
"""


import sys

from typing import Union, Optional, TextIO

from enum import Enum

import os

from io import IOBase 

from giant._typing import PATH, WriteableTarget


class REDIRECT(Enum):
    """
    An enumeration specifying the options of what to redirect
    """

    STDOUT = 'stdout'
    """
    Redirect STDOUT
    """

    STDERR = 'stderr'
    """
    Redirect STDERR
    """


class Tee:
    r"""
    This class support simultaneously writing to a file and stdout/stderr

    Essentially, all this does is serially call the write method for the original stdout/stderr object followed by the
    write method for the file being teed to.  This is also a context manager so you can use it with a ``with`` block for
    automatic opening/closing.  If not using in a with block, you should call :meth:`close` when done to set things back
    to normal.

    Example:

        >>> from giant.utilities.tee import Tee, REDIRECT
        >>> with Tee('mylog.txt', redirect=REDIRECT.STDOUT):
        ...     print('first line')
        ...     print('Second line')
        ...     print('I like tacos!!')
        first line
        second line
        I like tacos!!
        >>> with open('mylog.txt', 'r') as ifile:
        ...     print(ifile.readlines())
        ['first line\n', 'second line\n', 'I like tacos!!\n']

    Non-context manager example:

        >>> from giant.utilities.tee import Tee, REDIRECT
        >>> mytee = Tee('mylog.txt', redirect=REDIRECT.STDOUT)
        >>> print('first line')
        first line
        >>> print('second line')
        second line
        >>> print('I like tacos!!')
        I like tacos!!
        >>> mytee.close()
        >>> with open('mylog.txt', 'r') as ifile:
        ...     print(ifile.readlines())
        ['first line\n', 'second line\n', 'I like tacos!!\n']

    inspiration: https://stackoverflow.com/a/24583265/3431189
    """
    def __init__(self, file: Union[PATH, WriteableTarget], redirect: Union[REDIRECT, str] = REDIRECT.STDOUT, mode: str = "a+",
                 buff: int = -1):
        """
        :param file: The file to write to as a string or a ``WriteableTarget`` object which defines write, flush, and close
                     methods.
        :param redirect: Whether to redirect STDOUT or STDERR to the file.
        :param mode: The mode to open the file with.  Should be either 'w', 'a+', or 'r+'.  Ignored if ``file`` is
                     ``WriteableTarget``
        :param buff: How to buffer the file.  Should be -1 (for system default) or >= 1. Ignored if ``file`` is
                     ``WriteableTarget``
        """

        # grab the originals
        self._stdout: Optional[TextIO] = None 
        """
        A copy of the original STDOUT when ``redirect`` is set to ``STDOUT`` and this is open, otherwise ``None``
        """

        self._stderr: Optional[TextIO] = None 
        """
        A copy of the original STDERR when ``redirect`` is set to ``STDERR`` and this is closed, otherwise ``None``
        """

        self._file: Optional[WriteableTarget] = None 
        """
        The file object to tee to or ``None`` if this is closed.
        """

        if isinstance(file, WriteableTarget):
            self._file = file
        else:
            self._file = open(file, mode=mode, buffering=buff)

        self.redirect: REDIRECT = redirect if isinstance(redirect, REDIRECT) else REDIRECT(redirect.lower()) 
        """
        Specifies whether to tee STDOUT or STDERR to the file.
        """

        # store the appropriate WriteableTarget object and replace it
        if self.redirect == REDIRECT.STDOUT:
            self._stdout = sys.stdout
            sys.stdout = self
        elif self.redirect == REDIRECT.STDERR:
            self._stderr = sys.stderr
            sys.stderr = self

    def __repr__(self) -> str:
        return 'Tee({}, redirect={})'.format(repr(self.file), self.redirect)

    def __str__(self) -> str:
        if self._file is None:
            if self.redirect == REDIRECT.STDOUT:
                return "Closed Tee for STDOUT"
            else:
                return "Closed Tee for STDERR"
        else:
            if self.redirect == REDIRECT.STDOUT:
                return "Open Tee for STDOUT to {}".format(str(self._file))
            else:
                return "Open Tee for STDERR to {}".format(str(self._file))

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()
        
    @property
    def tty_destination(self) -> TextIO:
        """
        Returns the appropriate stdout or stderr depending on the selected REDIRECT option.
        
        :raises ValueError: If the tee has already been closed
        """
        
        match self.redirect:
            case REDIRECT.STDOUT:
                if self._stdout is None:
                    raise ValueError('It looks like you have called close before you should have')
                return self._stdout
            
            case REDIRECT.STDERR:
                if self._stderr is None:
                    raise ValueError('It looks like you have called close before you should have')
                return self._stderr
            
    @property
    def file(self) -> WriteableTarget:
        """
        Returns the writeable object to Tee to.
        
        :raises ValueError: if the Tee has already been closed
        """
        if self._file is None:
            raise ValueError('It looks like you have called close before you should have')
        return self._file

    def write(self, message: str) -> None:
        """
        This method dispatches to both the file write method and stdout or stderr depending on what was selected at
        initialization for the ``redirect`` parameter.

        :param message: The message to print
        :raises ValueError: If close method was called prior to this call
        """
        
        self.tty_destination.write(message)

        self.file.write(message)

    def flush(self) -> None:
        """
        This method flushes all data by calling both the file flush method and stdout or stderr's flush method depending
        on what was selected at initialization for the ``redirect`` parameter.

        :raises ValueError: If close method was called prior to this call
        """

        self.tty_destination.flush()
        self.file.flush()
        if hasattr(self.file, 'fileno'):
            os.fsync(getattr(self.file, 'fileno')())

    def close(self) -> None:
        """
        This method closes the file object and resets stdout/stderr to their values before creating an instance of this
        class.

        It is safe to call this multiple times, however, all subsequent calls will do nothing.
        """
        # reset STDOUT to what it was
        if self._stdout is not None:
            sys.stdout = self._stdout
            self._stdout = None

        # reset STDERR to what it was
        if self._stderr is not None:
            sys.stderr = self._stderr
            self._stderr = None

        # close file
        try:
            self.file.close()
            self._file = None
        except ValueError:
            pass


class DoubleTee:
    r"""
    This class provides the ability to simultaneously tee both STDOUT and STDERR to a file.

    This is done by creating two instances of the :class:`Tee` class, one for STDOUT and STDERR.

    This class can be used as a context manager, which will automatically handle closing the :class:`Tee` instances for
    you.  If you do not use it as a context manager, you should call :meth:`close` manually when you are done.

    Example:

        >>> from giant.utilities.tee import DoubleTee
        >>> import sys
        >>> with DoubleTee('mylog.txt'):
        ...     print('first line')
        ...     print('Second line')
        ...     print('I like tacos!!', file=sys.stderr)
        first line
        second line
        I like tacos!!
        >>> with open('mylog.txt', 'r') as ifile:
        ...     print(ifile.readlines())
        ['first line\n', 'second line\n', 'I like tacos!!\n']

    Non-context manager example:

        >>> from giant.utilities.tee import DoubleTee
        >>> import sys
        >>> doubletee = DoubleTee('mylog.txt')
        >>> print('first line')
        first line
        >>> print('second line')
        second line
        >>> print('I like tacos!!', file=sys.stderr)
        I like tacos!!
        >>> doubletee.close()
        >>> with open('mylog.txt', 'r') as ifile:
        ...     print(ifile.readlines())
        ['first line\n', 'second line\n', 'I like tacos!!\n']

    """

    def __init__(self, file: Union[PATH, WriteableTarget], mode: str = "a+", buff: int = -1):
        """
        :param file: The file to write to as a string or a ``WriteableTarget`` object which defines write, flush, and close
                     methods.
        :param mode: The mode to open the file with.  Should be either 'w', 'a', or 'r+'.  Ignored if ``file`` is
                     ``WriteableTarget``
        :param buff: How to buffer the file.  Should be -1 (for system default) or >= 1. Ignored if ``file`` is
                     ``WriteableTarget``
        """

        self.stdout: Tee = Tee(file, mode=mode, buff=buff, redirect=REDIRECT.STDOUT) 
        """
        An instance of Tee for teeing stdout to the requested file
        """

        self.stderr: Tee = Tee(self.stdout.file, redirect=REDIRECT.STDERR) 
        """
        An instance of Tee for teeing stderr to the requested file
        """

    def __repr__(self) -> str:
        return 'DoubleTee({})'.format(repr(self.stdout._file))

    def __str__(self) -> str:
        try:
            return "Open DoubleTee to {}".format(str(self.stdout.file))
        except ValueError:
            return "Closed DoubleTee"

    def __del__(self) -> None:
        """
        Deleter calls :meth:`close` in case it wasn't already called.
        """
        self.close()

    def __enter__(self) -> None:
        """
        Everything needed for entry is handled in ``__init__`` so does nothing
        """
        pass

    def __exit__(self, *args) -> None:
        """
        Dispatches to :meth:`close`
        """

        self.close()

    def close(self) -> None:
        """
        Closes both the STDOUT and STDERR Tees.
        """

        self.stdout.close()
        self.stderr.close()
