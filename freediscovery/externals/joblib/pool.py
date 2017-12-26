# -*- coding: utf-8 -*-

# Author: Olivier Grisel <olivier.grisel@ensta.org>
# Copyright: 2012, Olivier Grisel
# License: BSD 3 clause
# (adapted from joblib 0.11)

import os
import tempfile

from sklearn.externals.joblib.pool import SYSTEM_SHARED_MEM_FS

def _get_temp_dir(pool_folder_name, temp_folder=None):
    """Get the full path to a subfolder inside the temporary folder.

    This function is originally used in joblib.pool.MemmapingPool, but
    it could also be used in combination with joblib.Memory, provided
    that the produced cache folders are manually cleaned to avoid
    running out of memory.

    Parameters
    ----------
    pool_folder_name : str
        Sub-folder name used for the serialization of a pool instance.

    temp_folder: str, optional
        Folder to be used by the pool for memmaping large arrays
        for sharing memory with worker processes. If None, this will try in
        order:

        - a folder pointed by the JOBLIB_TEMP_FOLDER environment
          variable,
        - /dev/shm if the folder exists and is writable: this is a
          RAMdisk filesystem available by default on modern Linux
          distributions,
        - the default system temporary folder that can be
          overridden with TMP, TMPDIR or TEMP environment
          variables, typically /tmp under Unix operating systems.

    Returns
    -------
    pool_folder : str
       full path to the temporary folder
    use_shared_mem : bool
       whether the temporary folder is written to tmpfs
    """
    use_shared_mem = False
    if temp_folder is None:
        temp_folder = os.environ.get('JOBLIB_TEMP_FOLDER', None)
    if temp_folder is None:
        if os.path.exists(SYSTEM_SHARED_MEM_FS):
            try:
                temp_folder = SYSTEM_SHARED_MEM_FS
                pool_folder = os.path.join(temp_folder, pool_folder_name)
                if not os.path.exists(pool_folder):
                    os.makedirs(pool_folder)
                use_shared_mem = True
            except IOError:
                # Missing rights in the the /dev/shm partition,
                # fallback to regular temp folder.
                temp_folder = None
    if temp_folder is None:
        # Fallback to the default tmp folder, typically /tmp
        temp_folder = tempfile.gettempdir()
    temp_folder = os.path.abspath(os.path.expanduser(temp_folder))
    pool_folder = os.path.join(temp_folder, pool_folder_name)
    return pool_folder, use_shared_mem
