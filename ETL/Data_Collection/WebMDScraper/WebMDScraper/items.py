# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

from scrapy import Item, Field


class WebmdscraperItem(Item):
    Drug = Field()
    DrugId = Field()
    Condition = Field()
    Reviews = Field()
    Effectiveness = Field()
    EaseofUse = Field()
    Satisfaction = Field()
    Sides = Field()
    Sex = Field()
    Age = Field()
    UsefulCount = Field()
    Date = Field()
    #Indication = Field()
    #Type = Field()
    #Use = Field()


    #NumReviews = Field()