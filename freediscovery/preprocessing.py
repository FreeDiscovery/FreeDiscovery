import re



def emails_ignore_header(doc):
    """ Skip irrelevant email header fields """
    doc  = '\n'.join([line for line in doc.splitlines()
                      if not line.strip().startswith(('To:','From:','Date:', 'To:','Cc:','Sent:'))])
    return doc

processing_filters = {'emails_ignore_header': emails_ignore_header}
