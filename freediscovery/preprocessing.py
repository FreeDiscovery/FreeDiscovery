# Authors: Roman Yurchak
#
# License: BSD 3 clause

def emails_ignore_header(doc):
    """ Skip irrelevant email header fields """
    doc = '\n'.join([line for line in doc.splitlines()
                     if not line.strip().startswith(('To:', 'From:', 'Date:',
                                                     'Cc:', 'Sent:'))])
    return doc


processing_filters = {'emails_ignore_header': emails_ignore_header}
