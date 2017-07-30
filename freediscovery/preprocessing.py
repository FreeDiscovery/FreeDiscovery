import re



def emails_ignore_header(line):
    """ Skip irrelevant email header fields """
    if re.match('(?i)^\s*(?:To|From|Date|To|Cc|Sent):\s', line):
        return ''
    else:
        return line

processing_filters = {'emails_ignore_header': emails_ignore_header}
