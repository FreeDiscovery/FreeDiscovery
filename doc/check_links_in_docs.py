#!/usr/bin/python
# -*- coding: UTF-8 -*-

if __name__ == '__main__':

    import sys
    from urllib.request import urlopen
    from html.parser import HTMLParser

    BASE_URL = sys.argv[1]

    # List of all links to crawl through all pages
    links_to_crawl_through_all_pages = []

    # List of broken links in list [link url, page where it happened]
    broken_links = []

    class Page(HTMLParser):
        """ Check for broken links in the docs """
        def __init__(self, url_of_the_page, *args, **kwargs):
            # url of opened start page
            self.url_of_the_page = url_of_the_page
            super().__init__(*args, **kwargs)
            self.feed(self.read_site_content())
            self.crawl_through_all_pages()

        def read_site_content(self):
            return str(urlopen(self.url_of_the_page).read())

        def handle_starttag(self, tag, attrs):
            """ check whether the tag is a link tag
            to crawl through all the pages of Docs"""
            if tag == 'a':
                if attrs[0][1] == "reference internal":
                    # we verify the link and add validated one to list
                    if not self.validate(self.url_of_the_page + attrs[1][1]):
                        links_to_crawl_through_all_pages.append(
                            self.url_of_the_page + attrs[1][1])

        def validate(self, link):
            """ Checks whether or not to add a link to the list.
             The link is added to the list in the case:
                   1) It is not present in this list
                   2) There is not a javascript-code """
            return (link in links_to_crawl_through_all_pages or
                    'javascript:' in link)

        def crawl_through_all_pages(self):
            for address in links_to_crawl_through_all_pages:
                try:
                    PageWithLinks(address)
                except:
                    broken_links.append((address, self.url_of_the_page))

    class PageWithLinks(HTMLParser):
        """ Check for broken links in the docs """
        def __init__(self, url_of_the_page, *args, **kwargs):
            # list of all the links on reviewed page
            self.links_on_this_page = []
            # url of opened page
            self.url_of_the_page = url_of_the_page
            super().__init__(*args, **kwargs)
            self.feed(self.read_site_content())
            self.verify_links_on_page()

        def read_site_content(self):
            return str(urlopen(self.url_of_the_page).read())

        def handle_starttag(self, tag, attrs):
            if tag == 'a':
                if attrs[0][1] == "reference external":
                    link = str(attrs[1][1])
                    if link[0] == '.':
                        s = str(self.url_of_the_page)
                        if 'index.html' in s:
                            link = s[:s.find('index.html')] + link[2:]
                        else:
                            link = BASE_URL + link[2:]
                    if not self.validate(link):
                        self.links_on_this_page.append(link)

        def validate(self, link):
            """ The function checks whether or not to add a link to the list.
             The link is added to the list in the case:
                   1) It is not present in this list
                   2) There is not a javascript-code
            """
            return link in self.links_on_this_page or 'javascript:' in link

        def verify_links_on_page(self):
            for link in self.links_on_this_page:
                if '#' not in str(self.url_of_the_page):
                    try:
                        urlopen(link)
                    except:
                        broken_links.append((link, self.url_of_the_page))

    def print_list_of_broken_links():
        """ This function printes the list of broken links """
        print('')
        print('BROKEN LINKS:')
        i = 1
        for link in broken_links:
            print (str(i) + ') ' + link[0])
            print ('It happened on the page with URL: ' + link[1])
            i += 1
            print('')

    def write_list_with_broken_links_to_file():
        """ This function writes the list of broken links to file """
        f = open('list_of_URL_with_broken_links.txt', 'w')
        f.write('Broken links:\n\n')
        i = 1
        for link in broken_links:
            f.write(str(i) + ') ' + link[0] + '\n' +
                    'It happened on the page with URL: ' + link[1] + '\n\n')
            i += 1
        f.close()

    parser = Page(BASE_URL)
    print_list_of_broken_links()
    write_list_with_broken_links_to_file()
