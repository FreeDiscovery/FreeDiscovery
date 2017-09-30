# Authors: Denys Lazarenko
#          Roman Yurchak
#
# License: BSD 3 clause

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import numpy as np


def explain_categorization(weights, text, colormap):
    """Generate decorated HTML by adding <span> tags with background color to keywords depending on word's weight.

    Parameters
    ----------
    weights : dict[str, float]
        words as keys, weights (scores) as values
    text : str
        document's raw text
    colormap : matplotlib.colors.LinearSegmentedColormap
        color map used to decorate keywords with background color

    Returns
    -------
    str
        HTML string representing document's text decorated with <span> tags
    """
    try:
        from html import escape  # python 3.x
    except ImportError:
        from cgi import escape  # python 2.x
    from matplotlib.colors import Normalize

    text = escape(text)  # in order to have a valid HTML output after adding <span> tags

    max_val = max(abs(min(weights.values())), abs(max(weights.values())))

    norm = Normalize(vmin=-max_val, vmax=max_val)

    document_html_lines = list()
    for line in text.splitlines():
        line_decorated = _replace_with_color_spans(line, weights, colormap, norm)
        document_html_lines.append(line_decorated)
    return "<br/>".join(document_html_lines), norm


def _make_cmap(cmap_name='jet', alpha=0.2, filter_ratio=0.5):
    """Create a colormap which will be used to adding color spans to text

    Parameters
    ----------
    cmap_name : str
        name of colormap, see here for all possible values: http://matplotlib.org/users/colormaps.html
    alpha : float
        color's transparency
    filter_ratio : float
        make fully transparent (1 - filter_ratio) of the color bar

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        final color map with transparency
    """
    import matplotlib as mpl
    import matplotlib.cm as cm

    cmap = cm.get_cmap(cmap_name)

    # Extract colormap's colors and set new alpha
    cmap_array = cmap(np.arange(cmap.N))
    N = cmap_array.shape[0]
    if filter_ratio is not None:
        if not 0 <= filter_ratio <= 1:
            raise ValueError('filter_ratio = {} must be in the [0, 1] range'.format(filter_ratio))
        nf_ratio = 1 - filter_ratio
        cmap_array[:, -1] = 0.0
        cmap_array[:int(nf_ratio*N / 2), -1] = alpha
        cmap_array[-int(nf_ratio*N / 2):, -1] = alpha
    else:
        cmap_array[:, -1] = alpha


    # Create new colormap from the array with modified alpha
    cmap_with_trancparency = mpl.colors.ListedColormap(cmap_array)
    return cmap_with_trancparency


def _replace_with_color_spans(textline, weights, colormap, norm):
    """Given a line of text and a dictionary of word weights,
    add color span tag to those words in the textline that are present in the dictionary

    Parameters
    ----------
    textline : str
        text of one document's line
    weights : dict[str, float]
        words as keys, words' scores (weights) as values
    colormap : matplotlib.colors.LinearSegmentedColormap
        color map used to map the word's score to the background color of a <span>

    Returns
    -------
    str
        HTML string representing the word wrapped into a <span> tag with background color
    """

    html = textline
    positions = _get_keyword_positions(textline, weights.keys())
    positions_no_overlap = _keep_longest_overlapping_substrings(positions)


    # Perform replacement in the descreasing order of positions, i.e. from the end of the textline.
    # This way replacing words in the end of the string does not affect positions of words at the beginning
    for word_start, word_end in reversed(sorted(positions_no_overlap)):
        source_word = textline[word_start:word_end]
        # Using case-insensitive replacement while decorating with color spans.
        # Example:
        #   ERIC and Eric would be considered as the same word,
        # while keeping their original case.
        # We assume that dictionary has only lowercase words ('eric') in this case.
        key = source_word.lower()
        score = norm(weights[key])
        colored_word = _wrap_in_colored_span(source_word, score, colormap)
        # print(word_start, word_end, source_word, " ==> ", colored_word)
        html = html[:word_start] + colored_word + html[word_end:]
    return html


def _get_keyword_positions(textline, keywords):
    """Given a textline and a list of keywords, get a sorted list of positions of keywords found in the textline.

    Example
    -------
    textline: "Hello world, this is Eric."
    keywords: ["world", "eric"]
    would return [(6, 11), (21, 25)]. Each tuple represents word start and end index in the given string.

    Parameters
    ----------
    textline : str
        text of one document's line
    keywords : list[str]
        list of keywords for which we have scores (weights)

    Returns
    -------
    list[(int, int)]
        each tuple corresponds to a pair of (start, end) positions of a keyword inside a given line of text
    """
    positions_list = list()
    for word in keywords:
        if word in textline.lower():
            starts = [m.start() for m in re.finditer(word, textline.lower())]
            ends = [start+len(word) for start in starts]
            for start, end in zip(starts, ends):
                positions_list.append((start, end))
                # print(word, start, end)
    return sorted(positions_list)


def _keep_longest_overlapping_substrings(positions):
    """Deduplicate overlapping positions by keeping the longest possible keywords.

    Example
    -------
        document text contains string "email me at eric_bass@some-email.com"
        keywords vocabulary has "eric", "bass", "eric_bass"
        Thus keywords positions will have three tuples [(12, 16), (17, 21), (12, 21)].
        As we cannot overlap color spans, we have to deduplicate.
        In this examples we could either keep "eric" and "bass" or to keep "eric_bass" only and delete two others.
        We make a choice to keep the longest possible substring, deleting every other keyword overlapping with it.
        In this example the final output after deduplication will be [(12, 21].

    Example
    -------
        document text contains string "administrative procedures"
        keywords vocabulary has "administrative", "mini", "procedure", "pro"
        Thus keywords positions are [(0, 14), (2, 6), (15, 24), (15, 18)].
        Keeping longest possible keywords will eliminate "mini" and "pro" keywords,
        which were just accidentally matched and would represent a noise if kept.
        So, after deduplication only two tuples are left: [(0, 14), (15, 24)]

    Parameters
    ----------
    positions : list[(int, int)]
        each tuple corresponds to a pair of (start, end) positions of a keyword inside a given line of text

    Returns
    -------
    list[(int, int)]
        deduplicated version of keywords positions inside the text
    """
    overlap = True
    while overlap:
        overlap = False
        for idx in range(len(positions)-1):
            current_start, current_end = positions[idx]
            next_start, next_end = positions[idx+1]
            if current_end > next_start:
                overlap = True
                # In case of overlap, keep the longer substring,
                # and delete the indices of a shorter one from the list
                # Then break from the for-loop and restart analyzing the list
                if (current_end - current_start) > (next_end - next_start):
                    del positions[idx+1]
                else:
                    del positions[idx]
                break
    return sorted(positions)


def _wrap_in_colored_span(word, score, colormap):
    """Wrap a given word into an HTML <span> tag.

    Parameters
    ----------
    word : str
        word to be wrapped in a <span> tag with background color
    score : float
        word's score (weight)
    colormap : matplotlib.colors.LinearSegmentedColormap
        color map used to map the word's score to a background color of the span

    Returns
    -------
    str
        HTML tag <span> wrapped around the word to decorate it with a background color
    """
    rgba = _score_to_rgb(score, colormap)
    return '<span style="background-color: rgba({}, {}, {}, {});">{word}</span>'.format(*rgba, word=word)


def _score_to_rgb(score, colormap):
    """Convert word's score (weight) into an RGBa color.

    Parameters
    ----------
    score : float
        word's score (weight) from a dictionary
    colormap : matplotlib.colors.LinearSegmentedColormap
        color map used to map the word's score to an RGBa color

    Returns
    -------
    (int, int, int, float)
        RGBa representation of resulting color
    """
    mapped_rgba = colormap(score)
    r, g, b = [int(255*x) for x in list(mapped_rgba)[:3]]
    return r, g, b, mapped_rgba[-1]


def _create_random_weights(text, perc_keywords=0.5):
    """Create random weights for keywords.
    Keywords is a random subset of perc_keywords out of all words in the document.

    Parameters
    ----------
    text : str
        document text
    perc_keywords : float
        ratio of keywords of the whole vocabulary
    Returns
    -------
    dict[str: float]
        keywords as keys, their random weights as values
    """
    import numpy.random as rnd
    import random
    from sklearn.feature_extraction.text import CountVectorizer

    vect = CountVectorizer(input="content")
    _ = vect.fit([text])
    vocabulary = vect.vocabulary_
    nb_keywords = int(perc_keywords * len(vocabulary))
    keywords = random.sample(vocabulary.keys(), nb_keywords)
    features_weight = {word: rnd.random() for word in keywords}
    return features_weight


if __name__ == "__main__":
    fname = 'data/ds_001/raw/0.7.6.28635.txt'
    with open(fname) as in_file:  #, encoding='utf-8') as in_file:
        document_text = in_file.read().replace(u'\ufeff','')
    words_weights = _create_random_weights(document_text, 0.2)

    COLORMAP = _make_cmap()
    document_text_decorated = explain_categorization(words_weights, document_text, COLORMAP)
    with open('{}_decorated.html'.format(fname), 'w') as out_file:
        out_file.write(document_text_decorated)
