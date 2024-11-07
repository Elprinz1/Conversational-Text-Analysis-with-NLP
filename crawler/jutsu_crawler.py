import scrapy
from bs4 import BeautifulSoup


class BlogSpider(scrapy.Spider):
    name = 'narutospider'
    start_urls = ['https://naruto.fandom.com/wiki/Special:BrowseData/Jutsu']

    def parse(self, response):
        for href in response.css('.smw-columnlist-container')[0].css('a::attr(href)').extract():
            extracted_data = scrapy.Request(
                "https://naruto.fandom.com"+href, callback=self.parse_jutsu)
            yield extracted_data

        for next_page in response.css('a.mw-nextlink'):
            yield response.follow(next_page, self.parse)

    def parse_jutsu(self, response):
        jutsu_name = response.css(
            'span.mw-page-title-main::text').extract_first()
        jutsu_name = jutsu_name.strip()

        div_selector = response.css('div.mw-parser-output')[0]
        div_html = div_selector.extract()
        soup = BeautifulSoup(div_html, 'html.parser').find('div')

        if soup.find('aside'):
            aside = soup.find('aside')
            for row in aside.find_all('div', class_='pi-data'):
                if row.find('h3'):
                    if row.find('h3').text.strip() == 'Classification':
                        jutsu_type = row.find('div').text.strip()

        soup.find('aside').decompose()
        jutsu_description = soup.text.strip()
        jutsu_description = jutsu_description.split('Trivia')[0].strip()

        return dict(jutsu_name=jutsu_name,
                    jutsu_type=jutsu_type,
                    jutsu_description=jutsu_description)


# Load to file
# scrapy runspider crawler/jutsu_crawler.py -o data/jutsus.jsonl
