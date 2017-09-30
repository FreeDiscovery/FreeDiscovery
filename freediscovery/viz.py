# Authors: Roman Yurchak
#
# License: BSD 3 clause

from textwrap import dedent


def jupyter_hide_code():
    """ Allow to toggle display on notebooks """
    import IPython.core.display as di

    # This line will hide code by default when the notebook is exported as HTML
    di.display_html(dedent("""<script>
                           jQuery(function() {
                           if (jQuery("body.notebook_app").length == 0) {
                                jQuery(".input_area").toggle();
                                jQuery(".prompt").toggle();
                            }});
                            </script>"""), raw=True)

    # This line will add a button to toggle visibility of code blocks,
    # for use with the HTML export version
    di.display_html('''<button onclick="jQuery('.input_area').toggle(); jQuery('.prompt').toggle();">Toggle code</button>''', raw=True)
