# -*- coding: utf-8 -*-
import scrapy
from scrapy.cmdline import execute


class WorldSpiderSpider(scrapy.Spider):
    name = 'world_spider'
    allowed_domains = ['www.worldometers.info/coronavirus/']
    start_urls = ['http://www.worldometers.info/coronavirus//']

    def parse(self, response):
        country_dict = {}

        table = response.xpath('//table[contains(@class,"table table-bordered table-hover main_table_countries")]')
        trs = table.xpath('.//tr')[1:]
        for tr in trs:
            country = tr.xpath('.//td[1]//text()').extract_first()
            total_cases = tr.xpath('.//td[2]/text()').extract_first()
            new_cases = tr.xpath('.//td[3]/text()').extract_first()
            total_deaths = tr.xpath('.//td[4]/text()').extract_first()
            new_deaths= tr.xpath('.//td[5]/text()').extract_first()
            total_recovered = tr.xpath('.//td[6]/text()').extract_first()
            active_cases = tr.xpath('.//td[7]/text()').extract_first()
            serious_critical = tr.xpath('.//td[8]/text()').extract_first()
            tot_cases_1m_pop = tr.xpath('.//td[9]/text()').extract_first()
            tot_deaths_1m_pop = tr.xpath('.//td[10]/text()').extract_first()
            #country_dict[country] = [total_cases,new_cases,total_deaths,new_deaths,total_recovered,active_cases,serious_critical,tot_cases_1m_pop,tot_deaths_1m_pop]
            #print(country,total_cases,new_cases,total_deaths,new_deaths,total_recovered,active_cases,serious_critical,tot_cases_1m_pop,tot_deaths_1m_pop)

            yield {
                'country': country,
                'total_cases': total_cases,
                'new_cases': new_cases,
                'total_deaths':total_deaths,
                'new_deaths':new_deaths,
                'total_recovered':total_recovered,
                'active_cases':active_cases,
                'serious_critical':serious_critical,
                'tot_cases_1m_pop':tot_cases_1m_pop,
                'tot_deaths_1m_pop':tot_deaths_1m_pop
            }

    def execute_spider(self):
        execute(['scrapy','crawl','world_spider'])
#execute(['scrapy','crawl','world_spider'])