# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np


def explain_logistic_regression(weights, textlines):
    """
    Generate decorated HTML by adding <span> tags with background color to keywords depending on word's weight.

    Parameters
    ----------
    weights: [dict] {vocabulary : weights} with words [str] as keys, weights (scores) [float] as values
    textlines [list of str] list of text lines representing raw document text

    Returns
    -------
    [str] HTML string representing document's text decorated with <span> tags
    """
    document_html = ''
    for line in textlines:
        line = line.replace('<', '',).replace('>', '')  # in order to have a valid HTML output after adding <span> tags
        line_html = _replace_with_color_spans(line, weights)
        document_html += line_html + '<br/>'
    return document_html


def _make_cmap(cmap_name='jet', alpha=0.2):
    """
    Create a colormap which will be used to adding color spans to text
    """
    cmap = cm.get_cmap(cmap_name)

    # Extract colormap's colors and set new alpha
    cmap_array = cmap(np.arange(cmap.N))
    cmap_array[:, -1] = alpha

    # Create new colormap from the array with modified alpha
    cmap_with_trancparency = mpl.colors.ListedColormap(cmap_array)
    return cmap_with_trancparency


COLORMAP = _make_cmap()  # TODO another way to instanciate a colormap, not on the class level?


def _replace_with_color_spans(textline, weights):
    """
    Given a line of text and a dictionary of word weights,
    add color span tag to those words in the textline that are present in the dictionary
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
        score = weights[key]
        colored_word = _wrap_in_colored_span(source_word, score)
        print(word_start, word_end, source_word, " ==> ", colored_word)
        html = html[:word_start] + colored_word + html[word_end:]
    return html


def _get_keyword_positions(textline, keywords):
    """
    Given a textline and a list of keywords, get a sorted list of positions of keywords found in the textline.
    Example:
        textline: "Hello world, this is Eric."
        keywords: ["world", "eric"]
        would return [(6, 11), (21, 25)]. Each tuple represents word start and end index in the given string.
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
    """
    Deduplicate overlapping positions by keeping the longest possible keywords.

    Example 1:
        document text contains string "email me at eric_bass@some-email.com"
        keywords vocabulary has "eric", "bass", "eric_bass"
        Thus keywords positions will have three tuples [(12, 16), (17, 21), (12, 21)].
        As we cannot overlap color spans, we have to deduplicate.
        In this examples we could either keep "eric" and "bass" or to keep "eric_bass" only and delete two others.
        We make a choice to keep the longest possible substring, deleting every other keyword overlapping with it.
        In this example the final output after deduplication will be [(12, 21].

    Example 2:
        document text contains string "administrative procedures"
        keywords vocabulary has "administrative", "mini", "procedure", "pro"
        Thus keywords positions are [(0, 14), (2, 6), (15, 24), (15, 18)].
        Keeping longest possible keywords will eliminate "mini" and "pro" keywords,
        which were just accidentally matched and would represent a noise if kept.
        So, after deduplication only two tuples are left: [(0, 14), (15, 24)]
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


def _wrap_in_colored_span(word, score):
    rgba = _score_to_rgb(score, COLORMAP)
    return '<span style="background-color: rgba({}, {}, {}, {});">{word}</span>'.format(*rgba, word=word)


def _score_to_rgb(score, colormap):
    mapped_rgba = colormap(score)
    r, g, b = [int(255*x) for x in list(mapped_rgba)[:3]]
    return r, g, b, mapped_rgba[-1]


"""
import numpy.random as rnd
from sklearn.feature_extraction.text import CountVectorizer

def _create_weights(filename):
    vect = CountVectorizer(input="filename")
    _ = vect.fit([filename])
    features_weight = {word: rnd.random() for word in vect.vocabulary_}
    return features_weight

if __name__ == "__main__":
    fname = 'data/ds_001/raw/0.7.6.28635.txt'
    words_weights = _create_weights(fname)
    print(list(words_weights.items())[:3])

    with open(fname, encoding='utf-8') as in_file:
        document_lines = [line.strip() for line in in_file.readlines()]

    document_html = explain_logistic_regression(words_weights, document_lines)
    with open('{}_decorated.html'.format(fname), 'w') as out_file:
        out_file.write(document_html)
"""
